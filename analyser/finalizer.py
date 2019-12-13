import pymongo
import numpy as np
import textdistance

import analyser
from integration.db import get_mongodb_connection

currency_rates = {"RUB": 1.0, "USD": 63.72, "EURO": 70.59, "KZT": 0.17}


def get_attrs(document):
    attrs = document["analysis"]["attributes"]
    if document.get("user") is not None:
        attrs = document["user"]["attributes"]
    return attrs


def get_docs_by_audit_id(id: str, state, kind=None):
    db = get_mongodb_connection()
    documents_collection = db['documents']

    query = {
        'auditId': id,
        'parse.documentType': kind,
        "state": state
    }

    res = documents_collection.find(query).sort([("analysis.attributes.date.value", pymongo.ASCENDING)])
    return res


def save_violations(audit, violations):
    audit["violations"] = violations
    db = get_mongodb_connection()
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"violations": violations}})


def create_violation(document_id, founding_document_id, reference, violation_type):
    return {"document": document_id, "founding_document": founding_document_id, "reference": reference, "violation_type": violation_type}


def convert_to_rub(value_currency):
    value_currency["value"] = currency_rates.get(value_currency["currency"], "RUB") * value_currency["value"]
    value_currency["currency"] = "RUB"
    return value_currency


def get_constraints_rub(key, attributes):
    constraints = []
    for key2, value2 in attributes.items():
        if value2.get("parent") is not None and value2["parent"] == key:
            result = {}
            for key3, value3 in attributes.items():
                if value3.get("parent") is not None and value3["parent"] == key2:
                    if key3.endswith("sign"):
                        result["sign"] = value3["value"]
                    elif key3.endswith("value"):
                        result["value"] = value3["value"]
                    elif key3.endswith("currency"):
                        result["currency"] = value3["value"]
            constraints.append(result)
    for constraint in constraints:
        convert_to_rub(constraint)
    return constraints


def get_charter_diapasons(charter):
    #group by subjects
    subjects = {}
    charter_attrs = get_attrs(charter)
    min_constraint = np.inf
    for key, value in charter_attrs.items():
        if key.count("/") == 1:
            subject_type = value["value"]
            subject_map = subjects.get(subject_type)
            if subject_map is None:
                subject_map = {}
                subjects[subject_type] = subject_map
            constraints = get_constraints_rub(key, charter_attrs)
            for constraint in constraints:
                if subject_map.get(value["parent"]) is None:
                    subject_map[value["parent"]] = {"min": 0, "max": np.inf}
                if constraint["sign"] > 0:
                    if subject_map[value["parent"]]["min"] == 0:
                        subject_map[value["parent"]]["min"] = constraint["value"]
                    else:
                        old_value = subject_map[value["parent"]]["min"]
                        new_value = constraint["value"]
                        subject_map[value["parent"]]["min"] = min(old_value, new_value)
                    min_constraint = min(min_constraint, constraint["value"])
                else:
                    if subject_map[value["parent"]]["max"] == np.inf:
                        subject_map[value["parent"]]["max"] = constraint["value"]
                    else:
                        old_value = subject_map[value["parent"]]["max"]
                        new_value = constraint["value"]
                        subject_map[value["parent"]]["max"] = max(old_value, new_value)
    return subjects, min_constraint


def clean_name(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


def find_protocol(contract, protocols, org_level, audit):
    contract_attrs = get_attrs(contract)
    result = []
    for protocol in protocols:
        protocol_attrs = get_attrs(protocol)
        if protocol_attrs["org_structural_level"] == org_level:
            for protocol_key, protocol_value in protocol_attrs.items():
                if protocol_key.endswith("-name"):
                    for contract_key, contract_value in contract_attrs.items():
                        if contract_key.endswith("-name") and contract_value["value"] != audit["subsidiary"]["name"]:
                            clean_protocol_org = clean_name(protocol_value["value"])
                            clean_contract_org = clean_name(contract_value["value"])
                            distance = textdistance.levenshtein.normalized_distance(clean_contract_org, clean_protocol_org)
                            if distance > 0.9:
                                result.append(protocol)
    if len(result) == 0:
        return None
    else:
        return result[0]


def check_contract(contract, charters, protocols, audit):
    violations = []
    contract_attrs = get_attrs(contract)
    eligible_charter = None
    for charter in charters:
        charter_attrs = get_attrs(charter)
        if charter_attrs["date"]["value"] <= contract_attrs["date"]["value"]:
            eligible_charter = charter
            break

    if eligible_charter is None:
        violations.append(create_violation({"id": contract["_id"], "attribute": "date"}, None, None, "charter_not_found"))
        return violations
    else:
        charter_subject_map, min_constraint = get_charter_diapasons(eligible_charter)
        competences = charter_subject_map.get(contract_attrs["subject"]["value"])
        contract_value = convert_to_rub({"value": contract_attrs["sign_value_currency/value"]["value"], "currency": contract_attrs["sign_value_currency/currency"]["value"]})
        if competences is not None:
            eligible_protocol = None
            need_protocol_check = False
            for competence, constraint in competences.items():
                if constraint["min"] <= contract_value["value"] <= constraint["max"]:
                    need_protocol_check = True
                    eligible_protocol = find_protocol(contract, protocols, competence, audit)
            if eligible_protocol is not None:
                if eligible_protocol["analysis"]["attributes"]["date"]["value"] > contract_attrs["date"]["value"]:
                    violations.append(create_violation({"id": contract["_id"], "attribute": "date"},
                                           {"id": eligible_charter["_id"]},
                                           {"id": eligible_protocol["_id"], "attribute": "date"}, "contract_date_less_than_protocol_date"))
                else:
                    eligible_protocol_attrs = eligible_protocol["analysis"]["attributes"]
                    for key, value in eligible_protocol_attrs.items():
                        if key.endswidth("/value"):
                            converted_value = convert_to_rub({"value": value["value"], "currency": eligible_protocol_attrs[key[:-5] + "currency"]})
                            if min_constraint <= converted_value["value"] < contract_value["value"]:
                                violations.append(create_violation({"id": contract["_id"], "attribute": "sign_value_currency"},
                                                                   {"id": eligible_charter["_id"]},
                                                                   {"id": eligible_protocol["_id"], "attribute": eligible_protocol_attrs[key[:-6]]["parent"]}, "contract_value_great_than_protocol_value"))
            else:
                if need_protocol_check:
                    violations.append(create_violation({"id": contract["_id"], "attribute": "date"},
                                           {"id": eligible_charter["_id"]}, None, "protocol_not_found"))

    return violations


def finalize(audit):
    violations = []
    contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
    charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: k["analysis"]["attributes"]["date"]["value"])
    protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

    for contract in contracts:
        violations.extend(check_contract(contract, charters, protocols, audit))

    save_violations(audit, violations)


def create_fake_finalization(audit):
    violations = []
    contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
    charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: k["analysis"]["attributes"]["date"]["value"])
    protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

    for contract in contracts:
        contract_attrs = contract_attrs = contract["analysis"]["attributes"]
        eligible_protocol = next(protocols)
        eligible_protocol_attrs = eligible_protocol["analysis"]["attributes"]
        eligible_charter = charters[0]
        eligible_charter_attrs = eligible_charter["analysis"]["attributes"]
        violations.append(create_violation({"id": contract["_id"], "attribute": "date"}, None, None, "charter_not_found"))
        violations.append(create_violation({"id": contract["_id"], "attribute": "date"},
                                           {"id": eligible_charter["_id"]},
                                           {"id": eligible_protocol["_id"], "attribute": "date"}, "contract_date_less_than_protocol_date"))
        violations.append(create_violation({"id": contract["_id"], "attribute": "sign_value_currency"},
                                           {"id": eligible_charter["_id"]},
                                           {"id": eligible_protocol["_id"], "attribute": "agenda_item_1"},
                                           "contract_value_great_than_protocol_value"))
        violations.append(create_violation({"id": contract["_id"], "attribute": "date"},
                                           {"id": eligible_charter["_id"]}, None, "protocol_not_found"))
    save_violations(audit, violations)


if __name__ == '__main__':
    db = get_mongodb_connection()
    audits_collection = db['audits']
    audits = audits_collection.find({'status': 'Finalizing'}).sort([("createDate", pymongo.ASCENDING)])
    for audit in audits:
        finalize(audit)
        # create_fake_finalization(audit)
