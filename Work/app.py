#importing libraries
from shiny import App, render, ui,reactive
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns

MODEL_PATH = 'model_jlib'
NORM_PATH = 'scaler_jlib'
PCA_PATH = 'pca_jlib'

model = joblib.load(Path(__file__).parent / 'model_jlib')
normalizer = joblib.load(Path(__file__).parent / 'scaler_jlib')
pca = joblib.load(Path(__file__).parent/ 'pca_jlib')

#input data
df_rackets = pd.read_csv(Path(__file__).parent / "racket_data.csv")

#brand option
brand_ops = df_rackets["Manufacturer"].unique().tolist()
era_ops = df_rackets["new_old"].unique().tolist()

sns.set_theme()

# UI section starts from here 
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Tennis Racket Power Level & Recommendation
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(ui.input_slider("head_size", "Head size(sq. in)", 85, 125, np.mean([85, 125]), step = 1),
                         ui.input_slider("strung_weight", "Strung Weight (oz)", 8.2, 12.6, np.mean([8.2,12.6]),step = 0.1),
                         ui.input_slider("swing_weight", "Swing Weight", 227, 360, np.mean([227,360]),step = 1),
                         ui.input_slider("beam_width", "Beam Width(mm)", 17, 33, np.mean([17,33]),step = 0.5),
                         ui.input_slider("stiffness", "Flex", 47, 77, np.mean([47,77]),step = 1),
                         ui.input_action_button("btn", "Predict"),
                         ui.input_selectize("brand", "Brand", brand_ops, multiple=True),
                         ui.input_selectize("era", "Classic or Modern or Both", era_ops, multiple=True),
                         
                         ui.input_action_button("btn2", "Recommend"),
                        #  ui.output_plot("barchart"),
                         
                         ),
        
        ui.panel_main(ui.markdown(
        """
        ## Estimated Power Level
        """
    ),
                      ui.output_text_verbatim("txt", placeholder=True),
                    #   ui.output_table("table", placeholder=True),
                      ui.markdown(
        """
        ## Recommended Rackets
        """
    ),
                    #   ui.output_text_verbatim("txt", placeholder=True),
                      ui.output_table("table", placeholder=True)),
    ),
    
)
## server section 
def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.btn)
    
    
    
    def _():
        # Input data
        global Y_pred
        testset = pd.DataFrame([[input.head_size(),input.strung_weight(),input.swing_weight(),input.beam_width(),input.stiffness()]],
                               columns=['Head Size', 'Strung Weight', 'Swing Weight', 'Beam Width', 'Stiffness'],dtype=float)
        
        # normalize input 
        columns=['Head Size', 'Strung Weight', 'Swing Weight', 'Beam Width', 'Stiffness']
        X = testset[columns].values
        input_str = normalizer.transform(X)
        input_str = pca.transform(input_str)

        # getting prediction
        Y_pred = model.predict(input_str)

        Y_pred = Y_pred[0]
        
        if Y_pred == 0:
            pred = "Low"
        elif Y_pred == 1:
            pred = "Medium"
        else:
            pred = "High"
        
        
      
        # This output updates only when input.btn is invalidated.
        @output
        @render.text
        @reactive.event(input.btn)
        def txt():
            return f'The Specified Racquet has a power level of : {pred}'
        
    #racket recommendation system    
    @reactive.Effect
    @reactive.event(input.btn2)    
    
    
    def _():
        
        brand_list = list(input.brand())
        
        head = input.head_size()
        
        beam = input.beam_width()
        
        strungw = input.strung_weight()
        
        swingw = input.swing_weight()
        
        stiff = input.stiffness()
        
        power_level = Y_pred
        
        old_new = list(input.era())
     
        
        #function to keep top x rackets
        def more_than_x(a):
            x = 3
            if a["Name"].count() > x:
                return True
            else:
                return False
            
        #fuction to add data frame
        def concat(a,b, n, col):
            x = pd.concat([a,b[col].head(n)])
            return x
        
        #recommendation system
        def recommend(brand_list,head, strungw, swingw, power_level, stiff, old_new, beam):
            col_names = ['Name', 'Headsize (sq_inch)', 'Length (inn)', 'Strung Weight (oz)', 'Swingweight', 'Beamwidth', 'Stiffness', 'Stringing Pattern']

            num_of_rec = 3
            
            df = pd.DataFrame(columns= ['Name','Headsize (sq_inch)', 'Length (inn)', 'Strung Weight (oz)', 'Swingweight', 'Beamwidth', 'Stiffness', 'Stringing Pattern'])
            
            if len(old_new) == 2:
                first = df_rackets.loc[(df_rackets["Power Level"] == power_level)]
                
            elif old_new[-1] == "Modern":
                first = df_rackets.loc[(df_rackets["Power Level"] == power_level) & (df_rackets["new_old"] == "Modern")]
            elif old_new[-1] == "Classic":
                first = df_rackets.loc[(df_rackets["Power Level"] == power_level) & (df_rackets["new_old"] == "Classic")]
                
            for brand in brand_list:

                x = first.loc[(df_rackets["Manufacturer"] == brand) & (df_rackets["Beamwidth"].between(beam-0.7*2.78, beam+0.7*2.78))]
                    
                y = x.loc[x["Headsize (sq_inch)"].between(head-0.33*5.7, head+0.33*5.7)]
                z = y.loc[y["Strung Weight (oz)"].between(strungw-0.33*0.9, strungw+0.33*0.9)]
                a = z.loc[z["Swingweight"].between(swingw-0.2*16.6, swingw+0.2*16.6)]
                b = a.loc[a["Stiffness"].between(stiff-0.33*5.3, stiff+0.33*5.3)]
                
                if more_than_x(b):
                    # df = pd.concat([df, b[col_names].head(num_of_rec)])
                    df = concat(df,b, num_of_rec, col_names)
                elif more_than_x(a):
                    # df = pd.concat([df,a[col_names].head(num_of_rec)])
                    df = concat(df,a, num_of_rec, col_names)
                elif more_than_x(z):
                    # df = pd.concat([df,z[col_names].head(num_of_rec)])
                    df = concat(df,z, num_of_rec, col_names)
                elif more_than_x(y):
                    # df = pd.concat([df,y[col_names].head(num_of_rec)])
                    df = concat(df,y, num_of_rec, col_names)
                else:
                    # df = pd.concat([df,x[col_names].head(num_of_rec)])
                    df = concat(df,x, num_of_rec, col_names)
                
            return df
                    
        
        
        rec_result = recommend(brand_list,head,strungw,swingw,power_level, stiff, old_new, beam)
        

    
        # This output updates only when input.btn is invalidated.
        @output
        @render.table
        @reactive.event(input.btn2)
        
        def table():
            return rec_result

        
       
        #maybe add charts for visualisation later
        # @render.plot
        # def barchart():
        #     # note that input.traits() refers to the traits selected via the UI
        #     indx_trait = tennis_racket["Manufacturer"].isin(input.brand())
        #     indx_breed = tennis_racket["Power Level"].isin(input.power())
        #     # indx_pattern = tennis_racket["Stringing Pattern"].isin(input.pattern())
        #     indx_headLH = tennis_racket["HL or HH"].isin(input.HL_HH())
            
        #     # subset data to keep only selected traits and breeds
        #     sub_df = tennis_racket[indx_trait & indx_breed & indx_headLH]
        #     sub_df["dummy"] = 1

        #     # plot data. we use the same dummy value for x, and use hue to set
        #     # the bars next to eachother
        #     g = sns.catplot(
        #         data=sub_df, kind="bar",
        #         y="Power Level", x="dummy", hue="HL or HH",
        #         col="Manufacturer", col_wrap=3,
        #     )

        #     # remove labels on x-axis, which is on the legend anyway
        #     g.set_xlabels("")
        #     g.set_xticklabels("")
        #     g.set_titles(col_template="{col_name}")

        #     return g    
        
app = App(app_ui, server)