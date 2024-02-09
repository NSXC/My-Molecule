from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def generatesimilarmol(input_sequences, num_samples=10):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    inputs = tokenizer(input_sequences, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(input_ids=inputs.input_ids,
                              max_length=32,
                              num_return_sequences=num_samples,
                              early_stopping=False,
                              do_sample=True,
                              temperature=0.5)
    
    sim_mol = set()  
    i = 0
    while len(sim_mol) < num_samples and i < len(outputs):
        decoded_sequence = tokenizer.decode(outputs[i], skip_special_tokens=True)
        try:
            mol = Chem.MolFromSmiles(decoded_sequence)
            if mol is not None:
                mol_smiles = Chem.MolToSmiles(mol)
                if mol_smiles not in sim_mol: 
                    AllChem.Compute2DCoords(mol)
                    sim_mol.add(mol_smiles)
                    img_file = f"mol/generated_mol_{len(sim_mol)}.png"
                    Draw.MolToFile(mol, img_file)
                    print(f"Image saved: {img_file}")
        except Exception as e:
            print(f"Failed to process molecule: {decoded_sequence}, Error: {str(e)}")
        i += 1
    
    return [Chem.MolFromSmiles(smiles) for smiles in sim_mol]

# Current molecules for ADHD meds in SMILES format
samplemol = [
    "COC(=O)C(C1CCCCN1)C2=CC=CC=C2",
    "CC(CC1=CC=CC=C1)N",
    "C1=CC(=C(C(=C1)Cl)CC(=O)N=C(N)N)Cl",
    "CC(CC1=CC=CC=C1)NC(=O)C(CCCCN)N",
    "COC(=O)C(C1CCCCN1)C2=CC=CC=C2.Cl",
    "CC(CC1=CC=CC=C1)NC(=O)C(CCCCN)N",
    "CC1=CC=CC=C1OC(CCNC)C2=CC=CC=C2.Cl"
    ""
]

newmol = generatesimilarmol(samplemol)
print("Similar sequences:", newmol)
