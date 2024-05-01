import streamlit as st
from low_code import getdata,Expermint
x = getdata("mushroom_cleaned.csv")
e =Expermint()
y = e.setup(x, target="class", session_id=123)
print(y)
