from sklearn.model_selection import train_test_split
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem
import rdkit.Chem.Descriptors as dis
import torch as t
import torch.nn as n
import pandas as p
import numpy as np
def rd_kit_descriptors(smiles):

    smiles_string = str(smiles)

    mol = Chem.MolFromSmiles(smiles_string)

    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
                          'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
                          'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
                          'EState_VSA8', 'EState_VSA9', 'FpDensityMorgan1', 'FpDensityMorgan2',
                          'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                          'Ipc', 'Kappa1', 'Kappa2', 'Kappa3','fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    discreptor=MolecularDescriptorCalculator(chosen_descriptors).CalcDescriptors(mol)
    return t.FloatTensor(np.array(list(discreptor)))

smiles_list = [
    'CC', 'CCC', 'CCO', 'CCN', 'CCF', 'CCCl', 'CCBr', 'CCI', 'C=C', 'C#C', 'CC=CC', 'CC#CC', 'c1ccccc1', 'c1ccncc1',
    'c1cccnc1', 'c1ccc2ccccc2c1', 'c1ccc3c(c1)cc2ccccc23', 'CC(C)C', 'CC(C)(C)C', 'CC(C)O', 'CC(C)N', 'CC(C)F',
    'CC(C)Cl', 'CC(C)Br', 'CC(C)I', 'CC(C)C=C', 'CC(C)C#C', 'CC(C)c1ccccc1', 'CC(C)c1ccncc1', 'CC(C)c1cccnc1',
    'CC(C)c1ccc2ccccc2c1', 'CC(C)c1ccc3c(c1)cc2ccccc23', 'CCOC', 'CCN(C)C', 'CCS', 'CCP', 'CC(=O)O', 'CC(=O)N',
    'CC(=O)Cl', 'CC(=O)Br', 'CC(=O)I', 'CC(=O)C=C', 'CC(=O)C#C', 'CC(=O)c1ccccc1', 'CC(=O)c1ccncc1',
    'CC(=O)c1cccnc1', 'CC(=O)c1ccc2ccccc2c1', 'CC(=O)c1ccc3c(c1)cc2ccccc23', 'CCc1ccccc1', 'CCc1ccncc1',
    'CCc1cccnc1', 'CCc1ccc2ccccc2c1', 'CCc1ccc3c(c1)cc2ccccc23', 'CCOCC', 'CCNCC', 'CC(C)CC', 'CC(C)CO',
    'CC(C)CN', 'CC(C)CF', 'CC(C)CCl', 'CC(C)CBr', 'CC(C)CI', 'CC(C)C=CC', 'CC(C)C#CC', 'CC(C)c1ccccc1C',
    'CC(C)c1ccncc1C', 'CC(C)c1cccnc1C', 'CC(C)c1ccc2ccccc2C1', 'CC(C)c1ccc3c(c1)cc2ccccc23', 'CCC=C',
    'CCC#C', 'CCCCCC', 'CCCCCCO', 'CCCCCCN', 'CCCCCCF', 'CCCCCCCl', 'CCCCCCBr', 'CCCCCCI', 'CCCCCC=CC',
    'CCCCCC#CC', 'CCCCCCc1ccccc1', 'CCCCCCc1ccncc1', 'CCCCCCc1cccnc1', 'CCCCCCc1ccc2ccccc2c1', 'CCCCCCc1ccc3c(c1)cc2ccccc23',
    'CCC(O)C', 'CCC(N)C', 'CCC(F)C', 'CCC(Cl)C', 'CCC(Br)C', 'CCC(IC)C', 'CCC(C)=CC', 'CCC(C)=CC', 'CCC(C)#CC',
    'CCC(C)c1ccccc1', 'CCC(C)c1ccncc1', 'CCC(C)c1cccnc1', 'CCC(C)c1ccc2ccccc2c1', 'CCC(C)c1ccc3c(c1)cc2ccccc23',
    'CC(=O)OCC', 'CC(=O)NCC', 'CC(=O)ClCC', 'CC(=O)BrCC', 'CC(=O)ICC', 'CC(=O)C=CC', 'CC(=O)C#CC', 'CC(=O)c1ccccc1C',
    'CC(=O)c1ccncc1C', 'CC(=O)c1cccnc1C', 'CC(=O)c1ccc2ccccc2C1', 'CC(=O)c1ccc3c(c1)cc2ccccc23', 'CC(=O)CCc1ccccc1',
    'CC(=O)CCc1ccncc1', 'CC(=O)CCc1cccnc1', 'CC(=O)CCc1ccc2ccccc2c1', 'CC(=O)CCc1ccc3c(c1)cc2ccccc23', 'CC(=O)CCOC',
    'CC(=O)CCN', 'CC(=O)CCS', 'CC(=O)CCP', 'CC(=O)CCC=C', 'CC(=O)CCC#C', 'CC(=O)CCCCCC', 'CC(=O)CCCCCO',
    'CC(=O)CCCCCN', 'CC(=O)CCCCCF', 'CC(=O)CCCCCCCl', 'CC(=O)CCCCCCBr', 'CC(=O)CCCCCCI', 'CC(=O)CCCCCC=CC',
    'CC(=O)CCCCCC#CC', 'CC(=O)CCCCCCc1ccccc1', 'CC(=O)CCCCCCc1ccncc1', 'CC(=O)CCCCCCc1cccnc1', 'CC(=O)CCCCCCc1ccc2ccccc2c1',
    'CC(=O)CCCCCCc1ccc3c(c1)cc2ccccc23', 'CCCCCCCCCCCC', 'CCCCCCCCCCCCO', 'CCCCCCCCCCCCN', 'CCCCCCCCCCCCF',
    'CCCCCCCCCCCCCl', 'CCCCCCCCCCCCBr', 'CCCCCCCCCCCI', 'CCCCCCCCCCC=CC', 'CCCCCCCCCCC#CC', 'CCCCCCCCCCCc1ccccc1',
    'CCCCCCCCCCCc1ccncc1', 'CCCCCCCCCCCc1cccnc1', 'CCCCCCCCCCCc1ccc2ccccc2c1', 'CCCCCCCCCCCc1ccc3c(c1)cc2ccccc23',
    'CCCCCCCCCOC', 'CCCCCCCCCN', 'CCCCCCCCCCS', 'CCCCCCCCCCP', 'CCCCCCCCCCC(=O)O', 'CCCCCCCCCCC(=O)N', 'CCCCCCCCCCC(=O)Cl',
    'CCCCCCCCCCC(=O)Br', 'CCCCCCCCCCC(=O)I', 'CCCCCCCCCCC(=O)C=CC', 'CCCCCCCCCCC(=O)C#CC', 'CCCCCCCCCCC(=O)c1ccccc1',
    'CCCCCCCCCCC(=O)c1ccncc1', 'CCCCCCCCCCC(=O)c1cccnc1', 'CCCCCCCCCCC(=O)c1ccc2ccccc2c1', 'CCCCCCCCCCC(=O)c1ccc3c(c1)cc2ccccc23',
    'CCCCCCCCCCC(=O)CCc1ccccc1', 'CCCCCCCCCCC(=O)CCc1ccncc1', 'CCCCCCCCCCC(=O)CCc1cccnc1', 'CCCCCCCCCCC(=O)CCc1ccc2ccccc2c1',
    'CCCCCCCCCCC(=O)CCc1ccc3c(c1)cc2ccccc23', 'CCCCCCCCCCC(=O)CCOC', 'CCCCCCCCCCC(=O)CCN', 'CCCCCCCCCCC(=O)CCS',
    'CCCCCCCCCCC(=O)CCP', 'CCCCCCCCCCC(=O)CCC=C', 'CCCCCCCCCCC(=O)CCC#C', 'CCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCC',
    'CCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCC', 'CCCCCCCCCCCCCCCCCCCC'
]
def filter_smiles(smiles_list):
    valid_smiles = {}
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_weight = dis.MolWt(mol)
                valid_smiles[smiles] = mol_weight
        except:
            pass
    return valid_smiles

def create_data(smiles_list):
    valid_smiles = filter_smiles(smiles_list)

    X, Y = [], []
    for smiles, weight in valid_smiles.items():
        descriptors = rd_kit_descriptors(smiles)
        X.append(descriptors)
        Y.append(weight)
    return X, Y

class MyFirstNN(n.Module):
    def __init__(self, in_features=83, h1=400, h2=400, out_features=1):
        super(MyFirstNN, self).__init__()
        self.fc1 = n.Linear(in_features, h1)
        self.fc2 = n.Linear(h1, h2)
        self.out = n.Linear(h2, out_features)

    def forward(self, x):
        x = n.functional.relu(self.fc1(x))
        x = n.functional.relu(self.fc2(x))
        x = n.functional.relu(self.out(x))
        return x
def run(name):
    X_vals, Y_vals = create_data(smiles_list)
    X_num = np.array(X_vals)
    Y_num = np.array(Y_vals)
    X_train, X_test, y_train, y_test = train_test_split(X_num, Y_num, test_size=0.2, random_state=41)
    X_train = t.FloatTensor(X_train)
    X_test = t.FloatTensor(X_test)
    y_train = t.FloatTensor(y_train).view(-1, 1)
    y_test = t.FloatTensor(y_test).view(-1, 1)
    model = MyFirstNN(in_features=X_train.shape[1])
    loss_function = n.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 100000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    model.eval()
    with t.no_grad():
        test_output = model(X_test)
        test_loss = loss_function(test_output, y_test)
        print(f"Test Loss: {test_loss.item()}")

    t.save(model.state_dict(), f'{name}.pth')
