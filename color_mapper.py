import torch

def color_mapper (out : torch.Tensor , labels_dict : dict , num_classes):
    out = out.repeat(1,3,1,1)
    b,_,_,_ = out.size()
    batches = []
    for _ in range(b):
        r ,g, b = out[_]
        for i in range (num_classes):
            assert(str(i) in labels_dict ) ,f"class '{i}' not in dictionary"
            assert(i in out) , f"class '{i}' not in output"
            labels = labels_dict
            key = labels[str(i)]
            r =torch.where(r == i,key[0] ,r)
            g = torch.where(g == i ,key[1] ,g)
            b = torch.where(b== i ,key[2] ,b)
        out[_][0, : , :] = r
        out[_][1 ,:,:] = g
        out[_][2 , : , :] = b
        batches.append(out[_])
    return batches 
    
    