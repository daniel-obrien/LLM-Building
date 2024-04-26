import streamlit as st

# Streamlit app
def main():
    st.title("Translation App")
    
    # Container for input and output
    col1, col2 = st.columns(2)
    
    # Input text box
    with col1:
        st.subheader("Original Text")
        srclang = st.selectbox('Choose the source language :',
        ("English", "Turkish", "Polish", "German", "French", "Spanish"),
        index=None,
        help="Choose the source language", label_visibility="collapsed"
        )
        input_text = st.text_area("Input", height=300, max_chars=100, label_visibility="hidden")
        
    
    # Output text box
    with col2:
        st.subheader("Translated Text")
        trglang = st.selectbox('Choose the target language :',
        ("English", "Turkish", "Polish", "German", "French", "Spanish"),
        index=None,
        help="Choose the target language", label_visibility="collapsed"
        )
        translated_text = st.text_area("Output", height=300, max_chars=100, label_visibility="hidden")
    
    # Translate button
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    if st.button("Translate"):
        translated_text = input_text
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
