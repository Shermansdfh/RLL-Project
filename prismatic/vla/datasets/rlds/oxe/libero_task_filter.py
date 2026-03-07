"""Helper to get LIBERO task language instructions for dataset filtering.
Used when training on a subset of tasks (e.g. libero_task_ids="0" or "0,1").
"""

from typing import List, Set


def _grab_language_from_filename(task_name: str) -> str:
    """Convert task name (e.g. pick_up_the_black_bowl...) to human-readable language."""
    if task_name[0].isupper():
        if "SCENE10" in task_name:
            language = " ".join(task_name[task_name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(task_name[task_name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(task_name.split("_"))
    if ".bddl" in language:
        language = language[: language.find(".bddl")]
    return language


# Task names per suite (from LIBERO libero_suite_task_map)
LIBERO_TASK_NAMES = {
    "libero_spatial": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ],
    "libero_goal": [
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ],
    "libero_10": [
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ],
}


def get_libero_task_languages(suite_name: str, task_ids: List[int]) -> Set[str]:
    """
    Get the set of language instruction strings for the given task IDs.
    Used to filter the dataset when training on 1-2 tasks only.

    Args:
        suite_name: e.g. "libero_spatial", "libero_object", "libero_goal", "libero_10"
        task_ids: list of task indices (0-based)

    Returns:
        Set of lowercase language strings (e.g. {"pick up the black bowl from table center and place it on the plate"})
    """
    if suite_name not in LIBERO_TASK_NAMES:
        raise ValueError(
            f"Unknown libero suite '{suite_name}'. Must be one of {list(LIBERO_TASK_NAMES.keys())}"
        )
    task_names = LIBERO_TASK_NAMES[suite_name]
    result = set()
    for tid in task_ids:
        if tid < 0 or tid >= len(task_names):
            raise ValueError(f"task_id {tid} out of range [0, {len(task_names)}) for {suite_name}")
        lang = _grab_language_from_filename(task_names[tid])
        result.add(lang.lower().strip())
    return result
