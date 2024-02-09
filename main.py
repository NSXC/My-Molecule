from transformers import AutoTokenizer, AutoModelForCausalLM

def generatesimilarmol(input_sequences, num_samples=5):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    model = AutoModelForCausalLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    inputs = tokenizer(input_sequences, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(input_ids=inputs.input_ids,
                              max_length=32,
                              num_return_sequences=num_samples,
                              early_stopping=False,
                              do_sample=True,
                              temperature=0.7)
    
    sim_mol = []
    for output in outputs:
        decoded_sequence = tokenizer.decode(output, skip_special_tokens=True)
        sim_mol.append(decoded_sequence)

    return sim_mol

# Current molecules for ADHD meds in SMILES format
samplemol = [
    "COC(=O)C(C1CCCCN1)C2=CC=CC=C2",
    "CC(CC1=CC=CC=C1)N",
    "C1=CC(=C(C(=C1)Cl)CC(=O)N=C(N)N)Cl",
    "CC(CC1=CC=CC=C1)NC(=O)C(CCCCN)N",
    "COC(=O)C(C1CCCCN1)C2=CC=CC=C2.Cl"
]

newmol = generatesimilarmol(samplemol)
print("Similar sequences:", newmol)
