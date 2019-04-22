from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters


def add_shift(table, group_by=None, **params):
    check_required_parameters(_add_shift, params, ['table'])
    if group_by is not None:
        return _function_by_group(_add_shift, table, group_by=group_by, **params)
    else:
        return _add_shift(table, **params)

    
def _add_shift(table, input_col, shift_list, shifted_col=None, order_by=None, ordering='asc'):
     
    # always doing descending sort if ordering is not asc
    out_table = table.copy().sort_values(by=order_by, ascending=(ordering == 'asc')) if isinstance(order_by, list) and len(order_by) > 0 else table.copy() 
    
    if shifted_col is None:
        shifted_col = input_col
        
    for shift in shift_list:
        out_table['{shifted_col}_{shift}'.format(shifted_col=shifted_col, shift=shift)] = out_table[input_col].shift(shift)
        
    return {'out_table':out_table}
