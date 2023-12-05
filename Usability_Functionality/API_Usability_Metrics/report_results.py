import pandas as pd 
from AMNCI_metric import AMNCI
from AMONI_metric import AMONI
from DAI_metric import ADI_Methods
from APXI_metric import APXI
from AMGI_metric import compute_average_AMGI
import colorama
from colorama import Fore, Style
import argparse
from ast import literal_eval


def parse_args():
    parser = argparse.ArgumentParser(description='Report the results of the metrics')
    parser.add_argument('--data_path', help='The path to the data file', default='../Data/Documentation/output_init_torch.csv')
    return parser.parse_args()

def main():
    
    args = parse_args()
    csv_path = args.data_path
    df = pd.read_csv(csv_path)
    method_names = df['function_name'].tolist()


    AMGI_result = compute_average_AMGI(method_names)
    AMNCI_result = AMNCI(df)
    AMONI_result = AMONI(df)
    APXI_result = APXI(df)
    DAI_result = ADI_Methods(df)
    
    print("Tensorflow Documentation Metrics Results")
    # Report the results in a good way with colors
    print("AMGI Result:", Fore.GREEN + f"{AMGI_result*100:.2f}%" + Style.RESET_ALL)
    print("AMNCI Result:", Fore.GREEN + f"{AMNCI_result*100:.2f}%" + Style.RESET_ALL)
    print("AMONI Result:", Fore.GREEN + f"{AMONI_result*100:.2f}%" + Style.RESET_ALL)
    print("APXI Result:", Fore.GREEN + f"{APXI_result*100:.2f}%" + Style.RESET_ALL)
    print("DAI Result:", Fore.GREEN + f"{DAI_result*100:.2f}%" + Style.RESET_ALL)
    
    print("Pytorch Documentation Metrics Results")
    
    method_names_torch = df_torch['function_name'].tolist()
    AMGI_result = compute_average_AMGI(method_names_torch)
    AMNCI_result = AMNCI(df_torch)
    AMONI_result = AMONI(df_torch)
    APXI_result = APXI(df_torch)
    DAI_result = ADI_Methods(df_torch)
    
    print("AMGI Result:", Fore.GREEN + f"{AMGI_result*100:.2f}%" + Style.RESET_ALL)
    print("AMNCI Result:", Fore.GREEN + f"{AMNCI_result*100:.2f}%" + Style.RESET_ALL)
    print("AMONI Result:", Fore.GREEN + f"{AMONI_result*100:.2f}%" + Style.RESET_ALL)
    print("APXI Result:", Fore.GREEN + f"{APXI_result*100:.2f}%" + Style.RESET_ALL)
    print("DAI Result:", Fore.GREEN + f"{DAI_result*100:.2f}%" + Style.RESET_ALL)
    
    
    
    

    

if __name__ == "__main__":
    main()