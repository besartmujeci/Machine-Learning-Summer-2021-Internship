import streamlit as st
import recommender as rc
import model as md

st.title("Tapas Forum App")
option = st.sidebar.selectbox("Please select an option", ["", "Recommend",
                                                          "Classify"])
left_column, right_column = st.columns(2)
if option == "Recommend":
    left_column.write(":sunglasses:")
    user_input = st.text_input("Enter the text you would like"
                               " recommendations for.", "", )
    status = st.radio("Select # of Posts to Recommend: ", ("5", "10"))
    rec_button = st.button("Recommend Posts!")
    if rec_button:
        bar = st.progress(0)
        recs = rc.Recommender(user_input, status, bar).recommend()
        bar.progress(100)
        for rec in recs:
            st.write(rec)
elif option == "Classify":
    left_column.write(":sunglasses:")
    user_input1 = st.text_input("Enter the title of the text you would"
                                " like to classify.", "")
    user_input2 = st.text_area("Enter the body of the text you would"
                               " like to classify.", "", height=60)
    classify_button = st.button("Classify!")
    if classify_button:
        st.write(md.predict(user_input1 + user_input2))
