import torch as th

dict_list = [{'mean': th.tensor([1,1]), 'variance': th.tensor([0,1]), 'log_variance': th.tensor([0,1]), 
              'pred_xstart': th.tensor([0,1])}, 
              {'mean': th.tensor([0,2]), 'variance': th.tensor([0,1]), 'log_variance': th.tensor([0,1]),
                'pred_xstart': th.tensor([0,1])},
                        {'mean': th.tensor([0,1]), 'variance': th.tensor([0,1]),
                    'log_variance': th.tensor([0,1]), 'pred_xstart': th.tensor([0,1])}]

selected_idx_tensor = th.tensor([0,1]) 


def dict_list_to_dict_tensor(dict_list):
    out_dict = {}
    for key in dict_list[0].keys():
        print(key)
        value_list = []
        for dict in dict_list:
            value_list.append(dict[key])
        out_dict[key] = th.stack(value_list, dim=0)
    return out_dict

def select_tensor_by_idx(gradient_list_stack, selected_idx_tensor):
    selected_gradient_list = []
    for i, selected_idx in enumerate(selected_idx_tensor):
        selected_gradient_list.append(gradient_list_stack[i][selected_idx])
    selected_gradient_stack = th.stack(selected_gradient_list, dim=0)
    return selected_gradient_stack

print(select_tensor_by_idx(dict_list_to_dict_tensor(dict_list)['mean'], selected_idx_tensor))