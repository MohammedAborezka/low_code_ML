import streamlit as st
from low_code import getdata,Expermint
x = getdata("wine")
e =Expermint()
y = e.setup(x, target="wine", session_id=123)
print(y)
