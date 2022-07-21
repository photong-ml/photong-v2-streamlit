# Photong (Web App)
[![Streamlit app badge](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/leranjun/photong-web-app/main/app.py)

This is the front-end Streamlit app for Photong, an app that uses machine learning technology to generate a 16-bar melody from a photo.

## Deployment
The demo can be found [here](https://share.streamlit.io/leranjun/photong-web-app/main/app.py).

Deploying the Streamlit app locally requires Python 3.8 and [pipenv](https://pipenv.pypa.io/en/latest/).

Clone this project locally:
```
git clone https://github.com/leranjun/photong-web-app.git
cd photong-web-app
```

Then run:
```
pipenv install
pipenv run streamlit run app.py
```

### Deploying on Windows
You may need Microsoft C++ Build Tools to build some packages. For more information, follow [this Stack Overflow answer](https://stackoverflow.com/a/64262038).

## Usage
See [the step-by-step tutorial](https://github.com/leranjun/photong-web-app/blob/master/docs/README.md) for more information.
