from typing import List

def append_and_repad_list(list_of_items: List, item_to_append, pad_id):
    """
    Appends an item to a list and increases its size by 1, while ensuring padding consistency.

    Args:
        list_of_items (list): The original input list of items.
        item_to_append: The item to append to the list.
        pad_id: The padding ID used to fill the list to the desired size.

    Returns:
        list: A new list with the appended item, increasing the size of the input list by one.
    """
    # Remove all elements equal to the padding ID from the list
    items = [item for item in list_of_items if item != pad_id]
    
    # Append the new item to the filtered list
    items.append(item_to_append)
    
    # Ensure the result is "list_of_items size + 1", using pad_id for extra padding if necessary
    if len(items) < len(list_of_items) + 1:
        items += [pad_id] * (len(list_of_items) + 1 - len(items))

    return items
