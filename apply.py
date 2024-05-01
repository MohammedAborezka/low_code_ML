import streamlit as st
from low_code import getdata,Expermint
x = getdata("diabetes")
e =Expermint()
y = e.setup(x, target="diabetes", session_id=123)
print(y)
