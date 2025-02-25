import numpy as np
import sys

def action_mapper(action, param):
        """
        RL agent outputs actions in [-1,1]

        This function maps the action space to the actual values
        we split the action space (-1,1) into equal parts depending on the number of valid actions each parameter can take
        We then bin the action to the appropriate range
        """
        num_bins = len(param)
        action_bin = 2/num_bins
        
        # create boundries for each bin
        boundries = np.arange(-1, 1, action_bin)
        # find the index in the boundries array that the action falls in
        try:
            action_index = np.digitize(action , boundries) - 1
            #action_index = np.where(boundries <= round(action))[0][-1]
        except Exception as e:
            print(action)
      
        return action_index
    
def action_decoder_rl(act_encoded, rl_form):
        """
        Decode the action space for the RL agent
        """
        print("[Action Encoded]", act_encoded)
        act_decoded = {}
        
        # simle Encoding for string action space for memory controller
        pagepolicy = {0: "open", 1: "closed"}
        addressmapping = {0: "bank_row_col", 1: "row_bank_col", 2:"permutation", 3:"row_bank_col2", 4:"row_bank_col3"}
        #cas_mapper = {0:32, 1:64, 2:96}
        #cwl_mapper = {0:32, 1:64, 2:96}
        rcd_mapper = dict([(i, config) for i, config in enumerate(range(12, 17))])
        rp_mapper = dict([(i, config) for i, config in enumerate(range(12, 17))])
        ras_mapper = dict([(i, config) for i, config in enumerate(range(19, 40))])
        rrd_mapper = dict([(i, config) for i, config in enumerate(range(3, 6))])
        #faw_mapper = {0:52, 1:156, 2:104}
        #rfc_mapper = {0:1252, 1:1252, 2:1252}
        refi_mapper = dict([(i, config) for i, config in enumerate(range(4680, 468000, 46800))])


        if(rl_form == 'sa' or rl_form == 'macme_continuous'):
            print(action_mapper(act_encoded[4], ras_mapper))
            act_decoded["pagepolicy"] = pagepolicy[action_mapper(act_encoded[0], pagepolicy)]
            act_decoded["addressmapping"] = addressmapping[action_mapper(act_encoded[1], addressmapping)]
            act_decoded["rcd"]  =  rcd_mapper[action_mapper(act_encoded[2], rcd_mapper)]
            act_decoded["rp"]  =  rp_mapper[action_mapper(act_encoded[3], rp_mapper)]
            act_decoded["ras"]  =  ras_mapper[action_mapper(act_encoded[4], ras_mapper)]
            act_decoded["rrd"]  =  rrd_mapper[action_mapper(act_encoded[5], rrd_mapper)]
            act_decoded["refi"] = refi_mapper[action_mapper(act_encoded[6], refi_mapper)]
            
            #act_decoded["cas"] =  cas_mapper[action_mapper(act_encoded[0], cas_mapper)]
            #act_decoded["cwl"]  =  cwl_mapper[action_mapper(act_encoded[1], cwl_mapper)]
            #act_decoded["faw"]  =  faw_mapper[action_mapper(act_encoded[6],faw_mapper)]
            #act_decoded["rfc"]  =  rfc_mapper[action_mapper(act_encoded[7], rfc_mapper)]
        else:
            print("Invalid RL form")
            sys.exit()
        print("[Action Decoder]", act_decoded)
        return act_decoded
    
if __name__ == "__main__":
    
    
    #act_encoded = np.random.uniform(-1, 1, 7)
    ras = 0.5
    act_encoded = [0, 0 , 0, 0, ras, 0, 0]
    print(act_encoded)
    rl_form = 'sa'
    action_decoder_rl(act_encoded, rl_form)