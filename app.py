#importing libraries
from shiny import App, render, ui,reactive
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

MODEL_PATH = 'model_jlib'
NORM_PATH = 'scaler_jlib'
PCA_PATH = 'pca_jlib'

model = joblib.load(Path(__file__).parent / 'model_jlib')
normalizer = joblib.load(Path(__file__).parent / 'scaler_jlib')
pca = joblib.load(Path(__file__).parent/ 'pca_jlib')

# UI section starts from here 
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Tennis Racket Power Level 
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(ui.input_slider("head_size", "Head size(sq. in)", 85, 125, np.mean([85, 125]), step = 1),
                         ui.input_slider("strung_weight", "Strung Weight (oz)", 8.2, 12.6, np.mean([8.2,12.6]),step = 0.1),
                         ui.input_slider("swing_weight", "Swing Weight", 227, 360, np.mean([227,360]),step = 1),
                         ui.input_slider("beam_width", "Beam Width(mm)", 17, 33, np.mean([17,33]),step = 0.5),
                         ui.input_slider("stiffness", "Stiffness", 47, 77, np.mean([47,77]),step = 1),
                         ui.input_action_button("btn", "Predict"),
                         ),
        
        ui.panel_main(ui.markdown(
        """
        ## Model Output
        """
    ),
                      ui.output_text_verbatim("txt", placeholder=True),),
    ),
)
## server section 
def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.btn)
    
    def _():
        # Input data
    
        testset = pd.DataFrame([[input.head_size(),input.strung_weight(),input.swing_weight(),input.beam_width(),input.stiffness()]],
                               columns=['Head Size', 'Strung Weight', 'Swing Weight', 'Beam Width', 'Stiffness'],dtype=float)
        
        # normalize input 
        columns=['Head Size', 'Strung Weight', 'Swing Weight', 'Beam Width', 'Stiffness']
        X = testset[columns].values
        input_str = normalizer.transform(X)
        input_str = pca.transform(input_str)

        # getting prediction
        Y_pred = model.predict(input_str)
     
        if Y_pred[0] == 0:
            pred = "Low"
        elif Y_pred[0] == 1:
            pred = "Medium"
        else:
            pred = "High"
        
        
        
      
        # This output updates only when input.btn is invalidated.
        @output
        @render.text
        @reactive.event(input.btn)
        def txt():
            return f'The Estimated Power Level of the Specified Racket is: {pred}'
        
app = App(app_ui, server)