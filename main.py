import streamlit as st
from pricing_optimization import run_pricing_optimization
from utils import extract_python_code, extract_final_output
import matplotlib.pyplot as plt
import re

def main():
    st.set_page_config(page_title="Airline Pricing Optimization with Reflection AI Agent", layout="wide")
    st.title("Airline Pricing Optimization with Reflection AI Agent")

    st.sidebar.header("Market Data Input")
    average_price = st.sidebar.number_input("Average Price", value=100, step=1)
    competitor_prices = st.sidebar.text_input("Competitor Prices (comma-separated)", value="95,98,102,105")
    demand_forecast = st.sidebar.number_input("Demand Forecast", value=1000, step=10)
    current_inventory = st.sidebar.number_input("Current Seat Capacity", value=1200, step=10)

    api_choice = st.sidebar.selectbox("Choose API", options=["groq", "ollama"], index=0)

    if st.sidebar.button("Run Pricing Optimization"):
        market_data = {
            "average_price": average_price,
            "competitor_prices": [float(price.strip()) for price in competitor_prices.split(",")],
            "demand_forecast": demand_forecast,
            "current_inventory": current_inventory,
        }

        with st.spinner("Running pricing optimization..."):
            results, python_code, pricing_recommendation = run_pricing_optimization(market_data, api_choice)

        if results:
            st.success("Pricing optimization completed!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Optimization Process")
                for i, result in enumerate(results):
                    if result.type == "ai":
                        st.info(f"Generation Step {i + 1}:\n{result.content}")
                    elif result.type == "human":
                        st.warning(f"Reflection Step {i + 1}:\n{result.content}")

            with col2:
                st.subheader("Final Results")
                st.code(python_code, language="python")
                st.markdown(f"**Final Pricing Recommendation:**\n{pricing_recommendation}")

                # Visualize the pricing recommendation
                try:
                    recommended_price = float(re.search(r'\d+\.?\d*', pricing_recommendation).group())
                    prices = market_data['competitor_prices'] + [recommended_price]
                    labels = ['Competitor ' + str(i + 1) for i in range(len(market_data['competitor_prices']))] + ['Recommended']

                    fig, ax = plt.subplots()
                    ax.bar(labels, prices)
                    ax.set_ylabel('Price')
                    ax.set_title('Pricing Comparison')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                except:
                    st.error("Could not visualize the pricing recommendation.")

        else:
            st.error("Pricing optimization failed. Please check the logs for more information.")

if __name__ == "__main__":
    main()