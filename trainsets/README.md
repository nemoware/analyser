

## preparing trainset (several ugly steps):
1. Read all contracts (.doc, .docx) files from a dir,
 parse them and save into DB.  
This is done with `utilits.read_all_contracts main` or with 
 `utilits.make_subj_trianset`
 
2. Export JSON.  
run `../utilits/make_subj_trianset`, this wil produce `parsed_docs.json` -- a file
containig all pre-processed contracts from DB.

3.  Open  [contract subject trainset.ipynb](contract subject trainset.ipynb) and upload
`parsed_docs.json`.Run everything, it will write pickles on CSVs to Google Drive
 