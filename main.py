from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import math


def normalize_restaurant_name(restaurant_name: str) -> str:
    # Remove non-alphanumeric characters and convert to lowercase
    normalized = restaurant_name.lower()
    normalized = "".join(c for c in normalized if c.isalnum())
    return normalized


def fetch_restaurant_reviews_hash() -> Dict[str, List[str]]:
    reviews_hash = {}
    # - Read the file "./restaurant-data.txt", split it line by line
    with open("./restaurant-data.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        # Split on first period to separate restaurant name from review
        restaurant_name, review = line.split(". ", 1)
        restaurant_name = normalize_restaurant_name(restaurant_name)

        # Add to hash map - create new list if restaurant not seen before
        if restaurant_name not in reviews_hash:
            reviews_hash[restaurant_name] = []
        reviews_hash[restaurant_name].append(review)
    return reviews_hash


restaurant_reviews_hash = fetch_restaurant_reviews_hash()


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # This function takes in a restaurant name and returns the reviews for that restaurant.
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call.
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    reviews = restaurant_reviews_hash.get(
        normalize_restaurant_name(restaurant_name), []
    )
    return {restaurant_name: reviews}


def calculate_overall_score(
    restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]
) -> Dict[str, float]:
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service.
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.048}
    # NOTE: be sure to that the score includes AT LEAST 3  decimal places. The public tests will only read scores that have
    # at least 3 decimal places.

    N = len(food_scores)
    total_score = 0

    for i in range(N):
        food_score = food_scores[i]
        service_score = customer_service_scores[i]
        total_score += math.sqrt(food_score**2 * service_score)

    final_score = total_score * 10 / (N * math.sqrt(125))

    final_score = "{:.3f}".format(final_score)
    return {restaurant_name: final_score}


def get_data_fetch_agent_prompt() -> str:
    return "Identify the restaurant name from the given query and fetch the reviews for that restaurant using the fetch_restaurant_data function. Return the raw JSON results only."


def review_scoring_format() -> str:
    return "food_score = x, customer_service_score = y"


def get_review_analysis_agent_prompt() -> str:
    scoring_format = review_scoring_format()
    return f"Your are given a restaurant name and restaurant reviews. Extract the food score and customer service score for each review. The food score is a number between 1 and 5, and the customer service score is a number between 1 and 5. The food score is determined by the adjectives associated with the food, and the customer service score is determined by the adjectives associated with the customer service. Score 1/5 has one of these adjectives: awful, horrible, or disgusting. Score 2/5 has one of these adjectives: bad, unpleasant, or offensive. Score 3/5 has one of these adjectives: average, uninspiring, or forgettable. Score 4/5 has one of these adjectives: good, enjoyable, or satisfying. Score 5/5 has one of these adjectives: awesome, incredible, or amazing. Answer strictly with the following format, one line per review: `{scoring_format}`."


def get_scoring_agent_prompt() -> str:
    scoring_format = review_scoring_format()
    return f"Your are given a restaurant name and scores for the food and the customer service reviews, in the following format: `{scoring_format}`. Extract the food_scores and the customer_service_scores, then use the function `calculate_overall_score` and return the result only. Make sure that there are as many food scores than custoemr service scores."


# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = (
        "You must execute the functions and return the response."
    )
    # example LLM config for the entrypoint agent
    llm_config = {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}
        ]
    }
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config,
    )

    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=get_data_fetch_agent_prompt(),
        llm_config=llm_config,
    )

    review_analysis_agent = ConversableAgent(
        "review_analysis_agent",
        system_message=get_review_analysis_agent_prompt(),
        llm_config=llm_config,
    )

    scoring_agent = ConversableAgent(
        "scoring_agent",
        system_message=get_scoring_agent_prompt(),
        llm_config=llm_config,
    )

    # Unsure if should register_for_llm on the review_analysis_agent on the entrypoint_agent
    data_fetch_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant.",
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(
        fetch_restaurant_data
    )

    # Unsure if should register_for_llm on the review_analysis_agent on the entrypoint_agent
    scoring_agent.register_for_llm(
        name="calculate_overall_score",
        description="Calculate the overall score for a restaurant given food scores and customer service scores.",
    )(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(
        calculate_overall_score
    )

    # Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
    # If you decide to use another conversation pattern, feel free to disregard this code.

    # Uncomment once you initiate the chat with at least one agent.
    result = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": data_fetch_agent,
                "message": f"Query: {user_query}",
                "max_turns": 2,
                "summary_method": "last_msg",
            },
            {
                "recipient": review_analysis_agent,
                "message": "Here are the reviews",
                "max_turns": 2,
                "summary_method": "last_msg",
            },
            {
                "recipient": scoring_agent,
                "message": "Here are the scores",
                "max_turns": 2,
                "summary_method": "last_msg",
            },
        ]
    )
    return result


# DO NOT modify this code below.
if __name__ == "__main__":
    assert (
        len(sys.argv) > 1
    ), "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
