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
    parser.add_argument('--data', help='The path to the data file', default='../Data/Documentation/output_init_torch.csv')
    return parser.parse_args()


def main():
    
    args = parse_args()
    df = pd.read_csv(args.data)
    # df['']
    import pdb; pdb.set_trace()
    method_names = df['Name_Method'].tolist()
    # GET AMGI Metric
    AMGI_result = compute_average_AMGI(method_names)
    AMNCI_result = AMNCI(df)
    AMONI_result = AMONI(df)
    APXI_result = APXI(df)
    DAI_result = ADI_Methods(df)
    
    # Report the results in a good way with colors
    print("AMGI Result:", Fore.GREEN + f"{AMGI_result:.2f}%" + Style.RESET_ALL)
    print("AMNCI Result:", Fore.GREEN + f"{AMNCI_result:.2f}%" + Style.RESET_ALL)
    print("AMONI Result:", Fore.GREEN + f"{AMONI_result:.2f}%" + Style.RESET_ALL)
    print("APXI Result:", Fore.GREEN + f"{APXI_result:.2f}%" + Style.RESET_ALL)
    print("DAI Result:", Fore.GREEN + f"{DAI_result:.2f}%" + Style.RESET_ALL)

    

if __name__ == "__main__":
    main()