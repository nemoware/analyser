import pymongo
import numpy as np
import textdistance

from integration.db import get_mongodb_connection

currency_rates = {"RUB": 1.0, "USD": 63.72, "EURO": 70.59, "KZT": 0.17}

def get_audits():
  db = get_mongodb_connection()
  audits_collection = db['audits']

  res = audits_collection.find({'status': 'Finalizing'}).sort([("createDate", pymongo.ASCENDING)])
  return res

def remove_old_links(audit_id, contract_id):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    audit_collection.update_one({"_id": audit_id}, {"$pull": {"links": {"type": "analysis", "$or": [{"toId": contract_id}, {"fromId": contract_id}]}}})


def add_link(audit_id, doc_id1, doc_id2):
    db = get_mongodb_connection()
    audit_collection = db['audits']
    audit_collection.update_one({"_id": audit_id}, {"$push": {"links": {"fromId": doc_id1, "toId": doc_id2, "type": "analysis"}}})


def extract_text(span, words, text):
    first_idx = words[span[0]][0]
    last_idx = words[span[1]][0] - 1
    return text[first_idx:last_idx]


def get_nearest_header(headers, position):
    found_header = headers[0]
    for header in headers:
        if header["span"][0] > position:
            return found_header
        else:
            found_header = header
    return found_header


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
        "state": state,
        "$or": [
            {"analysis.attributes.date": {"$ne": None}},
            {"user.attributes.date": {"$ne": None}}
        ]
    }

    res = documents_collection.find(query).sort([("analysis.attributes.date.value", pymongo.ASCENDING)])
    docs = []
    for doc in res:
        docs.append(doc)
    return docs


def save_violations(audit, violations):
    db = get_mongodb_connection()
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"violations": violations}})
    db["audits"].update_one({'_id': audit["_id"]}, {"$set": {"status": "Done"}})


def create_violation(document_id, founding_document_id, reference, violation_type, violation_reason):
    return {"document": document_id, "founding_document": founding_document_id, "reference": reference, "violation_type": violation_type, "violation_reason": violation_reason}


def convert_to_rub(value_currency):
    value_currency["original_value"] = value_currency["value"]
    value_currency["original_currency"] = value_currency["currency"]
    value_currency["value"] = currency_rates.get(value_currency["currency"], "RUB") * float(value_currency["value"])
    value_currency["currency"] = "RUB"
    return value_currency


def get_max_value(doc_attrs):
    max_value = None
    sign = 0
    for key, value in doc_attrs.items():
        if key.endswith("/value"):
            if doc_attrs.get(key[:-5] + "sign") is not None:
                sign = doc_attrs[key[:-5] + "sign"]["value"]
            current_value = convert_to_rub({"value": value["value"], "currency": doc_attrs[key[:-5] + "currency"]["value"]})
            if max_value is None or max_value["value"] < current_value["value"]:
                max_value = current_value
    return max_value, sign

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
            subject_type = key.split("/")[1]
            subject_map = subjects.get(subject_type)
            if subject_map is None:
                subject_map = {}
                subjects[subject_type] = subject_map
            if subject_map.get(value["parent"]) is None:
                subject_map[value["parent"]] = {"min": 0, "max": np.inf, "competence_attr_name": key}
            constraints = get_constraints_rub(key, charter_attrs)
            if len(constraints) == 0:
                min_constraint = 0
            for constraint in constraints:
                if int(constraint["sign"]) > 0:
                    if subject_map[value["parent"]]["min"] == 0:
                        subject_map[value["parent"]]["min"] = constraint["value"]
                        subject_map[value["parent"]]["original_min"] = constraint["original_value"]
                        subject_map[value["parent"]]["original_currency_min"] = constraint["original_currency"]
                    else:
                        old_value = subject_map[value["parent"]]["min"]
                        new_value = constraint["value"]
                        if new_value < old_value:
                            subject_map[value["parent"]]["min"] = constraint["value"]
                            subject_map[value["parent"]]["original_min"] = constraint["original_value"]
                            subject_map[value["parent"]]["original_currency_min"] = constraint["original_currency"]
                    min_constraint = min(min_constraint, constraint["value"])
                else:
                    if subject_map[value["parent"]]["max"] == np.inf:
                        subject_map[value["parent"]]["max"] = constraint["value"]
                        subject_map[value["parent"]]["original_max"] = constraint["original_value"]
                        subject_map[value["parent"]]["original_currency_max"] = constraint["original_currency"]
                    else:
                        old_value = subject_map[value["parent"]]["max"]
                        new_value = constraint["value"]
                        if new_value > old_value:
                            subject_map[value["parent"]]["max"] = constraint["value"]
                            subject_map[value["parent"]]["original_max"] = constraint["original_value"]
                            subject_map[value["parent"]]["original_currency_max"] = constraint["original_currency"]
    if min_constraint == np.inf:
        min_constraint = 0
    return subjects, min_constraint


def clean_name(name):
    return name.replace(" ", "").replace("-", "").replace("_", "").lower()


def find_protocol(contract, protocols, org_level, audit):
    contract_attrs = get_attrs(contract)
    result = []
    for protocol in protocols:
        protocol_attrs = get_attrs(protocol)
        if protocol_attrs.get("org_structural_level") is not None and protocol_attrs["org_structural_level"]["value"] == org_level:
            for protocol_key, protocol_value in protocol_attrs.items():
                if protocol_key.endswith("-name"):
                    for contract_key, contract_value in contract_attrs.items():
                        if contract_key.endswith("-name") and contract_value["value"] != audit["subsidiary"]["name"]:
                            clean_protocol_org = clean_name(protocol_value["value"])
                            clean_contract_org = clean_name(contract_value["value"])
                            distance = textdistance.levenshtein.normalized_distance(clean_contract_org, clean_protocol_org)
                            if distance < 0.1:
                                result.append(protocol)
    if len(result) == 0:
        return None
    else:
        return result[0]


def check_contract(contract, charters, protocols, audit):
    violations = []
    contract_attrs = get_attrs(contract)
    contract_number = ""
    remove_old_links(audit["_id"], contract["_id"])
    if contract_attrs.get("number") is not None:
        contract_number = contract_attrs["number"]["value"]
    eligible_charter = None
    for charter in charters:
        charter_attrs = get_attrs(charter)
        if charter_attrs["date"]["value"] <= contract_attrs["date"]["value"]:
            eligible_charter = charter
            add_link(audit["_id"], contract["_id"], eligible_charter["_id"])
            break

    if eligible_charter is None:
        json_charters = []
        for charter in charters:
            charter_attrs = get_attrs(charter)
            json_charters.append({"id": charter["_id"], "date": charter_attrs["date"]["value"]})
        violations.append(create_violation({"id": contract["_id"], "number": contract_number, "type": contract["parse"]["documentType"]},
                                           None,
                                           None,
                                           "charter_not_found",
                                           {"contract": {"id": contract["_id"], "number": contract_number, "type": contract["parse"]["documentType"], "date": contract_attrs["date"]["value"]},
                                            "charters": json_charters}))
        return violations
    else:
        charter_subject_map, min_constraint = get_charter_diapasons(eligible_charter)
        eligible_charter_attrs = get_attrs(eligible_charter)
        competences = None
        if contract_attrs.get("subject") is not None:
            competences = charter_subject_map.get(contract_attrs["subject"]["value"])
        if competences is None:
            competences = charter_subject_map.get("Deal")
        contract_value = None
        if contract_attrs.get("sign_value_currency/value") is not None and contract_attrs.get("sign_value_currency/currency") is not None:
            contract_value = convert_to_rub({"value": contract_attrs["sign_value_currency/value"]["value"], "currency": contract_attrs["sign_value_currency/currency"]["value"]})
        if competences is not None and contract_value is not None:
            eligible_protocol = None
            need_protocol_check = False
            competence_constraint = None
            for competence, constraint in competences.items():
                if constraint["min"] <= contract_value["value"] <= constraint["max"]:
                    need_protocol_check = True
                    competence_constraint = constraint
                    eligible_protocol = find_protocol(contract, protocols, competence, audit)
                    if eligible_protocol is not None:
                        break

            attribute = None
            text = None
            min_value = None
            max_value = None
            if competence_constraint is not None:
                attribute = competence_constraint.get("competence_attr_name")
                if attribute is not None:
                    text = extract_text(eligible_charter_attrs[attribute]["span"],
                                        eligible_charter["analysis"]["tokenization_maps"]["words"],
                                        eligible_charter["analysis"]["normal_text"]) + "(" + get_nearest_header(eligible_charter["analysis"]["headers"], eligible_charter_attrs[attribute]["span"][0])["value"] + ")"
                if competence_constraint["min"] != 0:
                    min_value = {"value": competence_constraint["original_min"], "currency": competence_constraint["original_currency_min"]}
                if competence_constraint["max"] != np.inf:
                    max_value = {"value": competence_constraint["original_max"], "currency": competence_constraint["original_currency_max"]}

            contract_org2_type = None
            if contract_attrs.get("org-2-type") is not None:
                contract_org2_type = contract_attrs["org-2-type"]["value"]
            contract_org2_name = None
            if contract_attrs.get("org-2-name") is not None:
                contract_org2_name = contract_attrs["org-2-name"]["value"]

            if eligible_protocol is not None:
                add_link(audit["_id"], contract["_id"], eligible_protocol["_id"])
                eligible_protocol_attrs = get_attrs(eligible_protocol)
                protocol_structural_level = None
                if eligible_protocol_attrs.get("org_structural_level") is not None:
                    protocol_structural_level = eligible_protocol_attrs["org_structural_level"]["value"]

                if eligible_protocol_attrs["date"]["value"] > contract_attrs["date"]["value"]:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_number,
                         "type": contract["parse"]["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                        "contract_date_less_than_protocol_date",
                        {"contract": {"number": contract_number,
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_org2_type,
                                      "org_name": contract_org2_name},
                         "protocol": {"org_structural_level": protocol_structural_level,
                                      "date": eligible_protocol_attrs["date"]["value"]}}))
                else:
                    protocol_value, sign = get_max_value(eligible_protocol_attrs)
                    if protocol_value is not None:
                        if sign < 0 and min_constraint <= protocol_value["value"] < contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["parse"]["documentType"]},
                                {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                                "contract_value_great_than_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"]["value"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level, "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"], "currency": protocol_value["original_currency"]}}))

                        if sign == 0 and min_constraint <= protocol_value["value"] != contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["parse"]["documentType"]},
                                {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                                "contract_value_not_equal_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"]["value"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level, "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"], "currency": protocol_value["original_currency"]}}))

                        if sign > 0 and min_constraint <= protocol_value["value"] > contract_value["value"]:
                            violations.append(create_violation(
                                {"id": contract["_id"], "number": contract_number,
                                 "type": contract["parse"]["documentType"]},
                                {"id": eligible_charter["_id"],
                                 "date": eligible_charter_attrs["date"]["value"]},
                                {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                                "contract_value_less_than_protocol_value",
                                {"contract": {"number": contract_number,
                                              "date": contract_attrs["date"]["value"],
                                              "org_type": contract_org2_type,
                                              "org_name": contract_org2_name,
                                              "value": contract_attrs["sign_value_currency/value"]["value"],
                                              "currency": contract_attrs["sign_value_currency/currency"][
                                                  "value"]},
                                "protocol": {
                                     "org_structural_level": protocol_structural_level,
                                     "date": eligible_protocol_attrs["date"]["value"],
                                     "value": protocol_value["original_value"],
                                     "currency": protocol_value["original_currency"]}}))

            else:
                if need_protocol_check:
                    violations.append(create_violation(
                        {"id": contract["_id"], "number": contract_number,
                         "type": contract["parse"]["documentType"]},
                        {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                        {"id": eligible_charter["_id"], "attribute": attribute, "text": text},
                        {"type": "protocol_not_found", "subject": contract_attrs["subject"]["value"],
                         "org_structural_level": eligible_charter_attrs[eligible_charter_attrs[attribute]["parent"]]["value"],
                         "min": min_value,
                         "max": max_value
                         },
                        {"contract": {"number": contract_number,
                                      "date": contract_attrs["date"]["value"],
                                      "org_type": contract_org2_type,
                                      "org_name": contract_org2_name,
                                      "value": contract_attrs["sign_value_currency/value"]["value"],
                                      "currency": contract_attrs["sign_value_currency/currency"]["value"]}}))
    return violations


def finalize():
    audits = get_audits()
    for audit in audits:
        if audit["subsidiary"]["name"] == "Все ДО":
            print(f'.....audit {audit["_id"]} finalizing skipped')
            continue
        print(f'.....finalizing audit {audit["_id"]}')
        violations = []
        contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
        charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: get_attrs(k)["date"]["value"])
        protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

        for contract in contracts:
            violations.extend(check_contract(contract, charters, protocols, audit))

        save_violations(audit, violations)
        print(f'.....audit {audit["_id"]} is waiting for approval')


def create_fake_finalization(audit):
    violations = []
    contracts = get_docs_by_audit_id(audit["_id"], 15, "CONTRACT")
    charters = sorted(get_docs_by_audit_id(audit["_id"], 15, "CHARTER"), key=lambda k: get_attrs(k)["date"]["value"])
    protocols = get_docs_by_audit_id(audit["_id"], 15, "PROTOCOL")

    for contract in contracts:
        eligible_protocol = next(protocols)
        eligible_charter = charters[0]
        contract_attrs = get_attrs(contract)
        eligible_charter_attrs = get_attrs(eligible_charter)
        eligible_protocol_attrs = get_attrs(eligible_protocol)
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           None,
                                           None,
                                           "charter_not_found",
                                           {"contract": {"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"], "date": contract_attrs["date"]["value"]},
                                            "charters": [{"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]}]}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           "contract_date_less_than_protocol_date",
                                           {"contract": {"number": contract_attrs["number"]["value"], "date": contract_attrs["date"]["value"],
                                            "org_type": contract_attrs["org-2-type"]["value"], "org_name": contract_attrs["org-2-name"]["value"]},
                                            "protocol": {"org_structural_level": eligible_protocol_attrs["org_structural_level"]["value"], "date": eligible_protocol_attrs["date"]["value"]}}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           "contract_value_great_than_protocol_value",
                                           {"contract": {"number": contract_attrs["number"]["value"],
                                                         "date": contract_attrs["date"]["value"],
                                                         "org_type": contract_attrs["org-2-type"]["value"],
                                                         "org_name": contract_attrs["org-2-name"]["value"],
                                                         "value": contract_attrs["sign_value_currency/value"]["value"],
                                                         "currency": contract_attrs["sign_value_currency/currency"]["value"]},
                                            "protocol": {
                                                "org_structural_level": eligible_protocol_attrs["org_structural_level"][
                                                    "value"], "date": eligible_protocol_attrs["date"]["value"]}}))
        violations.append(create_violation({"id": contract["_id"], "number": contract_attrs["number"]["value"], "type": contract["parse"]["documentType"]},
                                           {"id": eligible_charter["_id"], "date": eligible_charter_attrs["date"]["value"]},
                                           {"id": eligible_charter["_id"], "attribute": "BoardOfDirectors/Deal-3",
                                            "text": extract_text(eligible_charter_attrs["BoardOfDirectors/Deal-3"]["span"], eligible_charter["analysis"]["tokenization_maps"]["words"], eligible_charter["analysis"]["normal_text"])},
                                           {"type": "protocol_not_found", "subject": contract_attrs["subject"]["value"], "org_structural_level": eligible_protocol_attrs["org_structural_level"]["value"],
                                            "min": {"value": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/value"]["value"],
                                            "currency": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/currency"]["value"]},
                                            "max": {"value": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/value"]["value"],
                                                    "currency": eligible_protocol_attrs["agenda_item_1/sign_value_currency-2/currency"]["value"]}
                                            },
                                           {"contract": {"number": contract_attrs["number"]["value"],
                                                         "date": contract_attrs["date"]["value"],
                                                         "org_type": contract_attrs["org-2-type"]["value"],
                                                         "org_name": contract_attrs["org-2-name"]["value"],
                                                         "value": contract_attrs["sign_value_currency/value"]["value"],
                                                         "currency": contract_attrs["sign_value_currency/currency"]["value"]}}))
    save_violations(audit, violations)


if __name__ == '__main__':
    finalize()

