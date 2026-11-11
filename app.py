from functools import partial
from shiny import App, ui, render
from shiny.ui import page_navbar
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from shiny.render import DataGrid
from imblearn.over_sampling import SMOTENC
import kagglehub
import os
import seaborn as sns
from matplotlib.pyplot import subplots
from sklearn import tree
from sklearn.tree import plot_tree

path = kagglehub.dataset_download("houcembenmansour/predict-diabetes-based-on-diagnostic-measures")
files = os.listdir(path)
df = pd.read_csv(os.path.join(path, files[0]))
df_before_cleaning = df.copy()
# Cleaning
# Converting comma-decimal numbers to float
df["chol_hdl_ratio"] = pd.to_numeric(
    df["chol_hdl_ratio"].astype(str).str.replace(",", "."),
    errors="coerce"
)

df["bmi"] = pd.to_numeric(
    df["bmi"].astype(str).str.replace(",", "."),
    errors="coerce"
)

df["waist_hip_ratio"] = pd.to_numeric(
    df["waist_hip_ratio"].astype(str).str.replace(",", "."),
    errors="coerce"
) 
# cleaning text 
df["gender"] = df["gender"].astype(str).str.strip().str.lower()
df["diabetes"] = df["diabetes"].astype(str).str.strip().str.lower()

# Variable mapping
df["diabetes"] = df["diabetes"].map({"no diabetes":0,"diabetes":1})
df["gender"] = df["gender"].map({"female":0,"male":1})

df.drop(columns=['patient_number'], inplace=True)


# Train - test set definition
X = df.drop(columns=["diabetes"])
y = df["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=287
)
# Scaling w mean and standard dev

# Spliting numeric columns to scale it
num_cols = X_train.select_dtypes(include="number").columns.drop("gender", errors="ignore")

scaler = StandardScaler()

X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])

# Oversampling since sataset is imbalanced
gender_index = X_train.columns.get_loc("gender") # Find column index of gender

# Applying SMOTENC since both categorical and numerical features present
X_train_resampled, y_train_resampled = SMOTENC(
    categorical_features=[gender_index],
    random_state=287
).fit_resample(X_train_sc, y_train)

# ---------- Decision Tree Cross Validation and Fitting -------------
alpha_parameters = {"ccp_alpha": [1,0.1,0.01,0.001,0.0001]}
treeCV = GridSearchCV(DecisionTreeClassifier(random_state=67), alpha_parameters, scoring='accuracy',cv=10)
treeCV.fit(X_train_resampled, y_train_resampled)

#fitting decision tree model
dt_model = DecisionTreeClassifier(ccp_alpha=0.01,random_state=67).fit(X_train_resampled,y_train_resampled)
y_pred_dt = dt_model.predict(X_test_sc)

# calculationg probabilities for roc-auc later
y_proba = dt_model.predict_proba(X_test_sc)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

alpha_grid = [p["ccp_alpha"] for p in treeCV.cv_results_["params"]]


# ------------- Random Forset Cross Validation and Fitting ------------
feature_parameters = {"max_features": [2,4,6,8,10,12,14]}
rfCV_feature = GridSearchCV(RandomForestClassifier(n_estimators=300,bootstrap=True,oob_score=True,random_state=67), feature_parameters, scoring='accuracy',cv=10)
rfCV_feature.fit(X_train_resampled, y_train_resampled)

#fitting random forest model
rf_model = RandomForestClassifier(n_estimators=300,max_features=2,bootstrap=True,oob_score=True,random_state=67).fit(X_train_resampled,y_train_resampled)
y_pred_rf = rf_model.predict(X_test_sc)

# calculating probabilities for roc-auc later
y_proba_rf = rf_model.predict_proba(X_test_sc)[:, 1]

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)


feature_grid = [p["max_features"] for p in rfCV_feature.cv_results_["params"]]


# ------------------ Gradient Boosting Cross Validation and Fitting ---------------

# cross validation for learning_rate

grd = GradientBoostingClassifier(max_depth=3, n_estimators=200, random_state=67)
learning_parameters = {"learning_rate": [0.001,0.01,0.1,1]}
grdCV = GridSearchCV(grd, learning_parameters, scoring='accuracy',cv=10)
grdCV.fit(X_train_resampled, y_train_resampled)

#fitting gradient boosting model
grd_model = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=200, random_state=67).fit(X_train_resampled,y_train_resampled)

y_pred_grd = grd_model.predict(X_test_sc)

# calculationg probabilities for roc-auc later
y_proba_grd = grd_model.predict_proba(X_test_sc)[:, 1]

fpr_grd, tpr_grd, threshold_grd = roc_curve(y_test, y_proba_grd)
auc_grd = roc_auc_score(y_test, y_proba_grd)

learning_grid = [p["learning_rate"] for p in grdCV.cv_results_["params"]]


app_ui = ui.page_fluid(

    # ---------- Global Styling ----------
    ui.tags.style("""
        body { font-size: 18px; } /* main body text */
        .navbar {
            border-bottom: 1.5px solid #e0e0e0;
        }

        .nav-link.active {
            border-bottom: 3px solid #2b7cff;
        }
                
        .navbar-brand { /* main page title */
            font-size: 30px; 
            margin-right: 60px;
        }

        .nav-link {  /* tab names */
            font-size: 16px; 
        }
                  
        h4 {  /* inside title of each page */
            font-size: 26px;   
            font-weight: 700;
        }
                  
        .card {  /* distance between each card */
            margin-bottom: 30px;
        }
    """),

    # ---------- Navbar ----------
    ui.page_navbar( 

        # ---------- Page 1 ----------
        ui.nav_panel(
            "Data",
            ui.h4("Background About the Data"), #main title of the "PAGE"
            # -------- First Line of Cards for Data --------
            ui.layout_columns(

                ui.card(
                    ui.card_header(ui.tags.span("Data and Task", style="font-size:20px; font-weight:500;")
                    ),
                    ui.p(
                        ui.HTML( #html utilized since it allows easy controll of text output
                            "<br>"
                            "In today's age, diabetes is one of the major global health challenges. According to the World Health Organisation, 1.6 million deaths were caused by diabetes in 2021. Studies emphasise that approximately %33 of cases remain undiagnosed, despite the benefits of early detection.<br><br> "
                            "The dataset from Kaggle contains 390 observations and 15 features describing patients’ health characteristics, with diabetes diagnosis as the target variable.<br><br>"
                            "Tree-based models (Decision Tree, Random Forest, and Gradient Boosting) were developed to predict diabetes diagnosis, using preprocessing, hyperparameter tuning, and performance evaluation with standard classification metrics."
                        )
                    )
                ),
                ui.card(
                    ui.card_header(ui.tags.span("Quick View of Data", style="font-size:20px; font-weight:500;")
                    ),
                    ui.output_data_frame("df_head_table")
                ),
            col_widths=[4,8]   
            ),
            # -------- Second Card for Hue Plot--------
            ui.card(
                    ui.card_header(ui.tags.span("How Health Measurements Relate to Diabetes", style="font-size:20px; font-weight:500;")
                    ),
                    ui.input_selectize(
                        "vars_hue_plot",
                        "Select at Least Two Features:",
                        choices=[
                            "cholesterol",
                            "glucose",
                            "hdl_chol",
                            "chol_hdl_ratio",
                            "age",
                            "height",
                            "bmi",
                            "systolic_bp",
                            "diastolic_bp",
                            "waist",
                            "hip",
                            "waist_hip_ratio",
                        ],
                        multiple=True, #can select multiple
                        selected=[      #default selected when opened
                            "age",
                            "height",
                            "bmi",
                            "cholesterol"
                            
                        ]
                    ),
                ui.hr(), #line seperator
                ui.output_plot("eda_hue_plot"),
                full_screen=True
            ),
            # -------- Third Card for Distribution Plot--------
            ui.card(
                ui.card_header(ui.tags.span("How Health Measures Differ Between People With and Without Diabetes", style="font-size:20px; font-weight:500;")
                ),
                ui.layout_columns(
                    ui.input_selectize( #choose features other than gender
                        "var_dist_plot",
                        "Select a Feature: ",
                        choices=[ 
                        "cholesterol",
                        "glucose",
                        "hdl_chol",
                        "chol_hdl_ratio",
                        "age",
                        "height",
                        "bmi",
                        "systolic_bp",
                        "diastolic_bp",
                        "waist",
                        "hip",
                        "waist_hip_ratio",
                        ],
                        multiple=False, #single selection
                        selected="age"
                    ),
                    ui.input_selectize( #choose gender
                        "dist_gender_mode",
                        "Choose a Gender Mode:",
                        choices=[
                            "On",
                            "Off",
                        ],
                        multiple=False,
                        selected="On"
                    ),
                ),  
                ui.hr(),  #line seperator
                ui.output_plot("eda_dist_plot"),
                full_screen=True # ability to make the plot outputs full screen
            ),
            # -------- Fourth Card for Box Plot--------
            ui.card(
                ui.card_header(ui.tags.span("How Health Measures Differs by Diabetes and Gender", style="font-size:20px; font-weight:500;")
                ),
                ui.layout_columns(
                    ui.input_selectize( #choose features other than gender
                        "var_box_plot",
                        "Select a Feature: ",
                        choices=[   
                            "cholesterol",
                            "glucose",
                            "hdl_chol",
                            "chol_hdl_ratio",
                            "age",
                            "height",
                            "bmi",
                            "systolic_bp",
                            "diastolic_bp",
                            "waist",
                            "hip",
                            "waist_hip_ratio",
                        ],
                        multiple=False,
                        selected="age"
                    ),
                    ui.input_selectize( #choose gender
                        "box_gender_mode",
                        "Choose a Gender Mode:",
                        choices=[ 
                            "On",
                            "Off",
                        ],
                        multiple=False,
                        selected="On"
                    ),
                ),    
                ui.hr(),   #line seperator 
                ui.output_plot("eda_box_plot"),
                full_screen=True
            ),
            # -------- Fifth Card for Heat Map Plot--------
            ui.card(

                ui.card_header(ui.tags.span("Correlation Between Health Measurements", style="font-size:20px; font-weight:500;")
                ),

                ui.input_selectize(
                    "vars_heat_map",
                    "Select at Least Two Features:",
                    choices=[
                    "cholesterol",
                    "glucose",
                    "hdl_chol",
                    "chol_hdl_ratio",
                    "age",
                    "height",
                    "bmi",
                    "systolic_bp",
                    "diastolic_bp",
                    "waist",
                    "hip",
                    "waist_hip_ratio",
                    ],
                    multiple=True,
                    selected=[
                    "age",
                    "bmi",
                    "cholesterol",
                    "height"
                    ]
                ),
                ui.hr(), #line seperator
                ui.output_plot("heat_map_plot"),

                full_screen=True
            )
        ),

        # ---------- Page 2 ----------
        ui.nav_panel(
            "Hyperparameter Tuning",

            ui.h4("Finding the Best Model Settings"),

            ui.card(
                
                ui.p(
                    ui.HTML(
                        "Hyperparameters are model settings that control how a machine learning model learns from the data. Hyperparameter tuning is essential for improving predictive performance and preventing the model from becoming too complex. "
                        "Tuned parameters are the tree complexity level (ccp_alpha) for Decision Trees, the number of features considered at each split (max_features) for Random Forests, and the learning rate (learning_rate) for Gradient Boosting. <br><br>"
                        "Below, you can adjust each parameter and see how the model’s performance changes with different parameter values for each tree-based model."
                    )

                )
            ),     

            # -------- First Card --------
            ui.card(
                ui.card_header(
                    ui.tags.span("Decision Tree", style="font-size:20px; font-weight:500;")
                ),

                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_slider("alpha", ui.HTML("cpp_alpa:<br>Complexity level of the tree"), 
                            min=0,
                            max=len(alpha_grid) - 1,
                            value=4,
                            step=1,
                            ticks=False,
                            animate=True
                        ),
                        ui.output_text("alpha_value"),
                        bg="#f8f8f8",
                        width="25%"
                        
                    ),

                ui.output_plot("dt_alpha_plot"),
                ui.output_ui("dt_alpha_text"),
                full_screen=True
                ),
            ),

            # -------- Second Card --------
            ui.card(
                ui.card_header(ui.tags.span("Random Forest", style="font-size:20px; font-weight:500;")
                ),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_slider("max_features", ui.HTML("max_features:<br>Number of features considered at each split"), #html used for being able to control title of the slider
                            min=2,
                            max=14,
                            step=2, 
                            value=2,
                            ticks=True,
                            animate=True
                        ),
                        ui.output_text("max_features_value"),
                        bg="#ecf6fd",
                        width="25%",
                    ),
                
                ui.output_plot("rf_features_plot"),
                ui.output_ui("rf_features_text"), #output_ui used since html in server
                full_screen=True
                ),
            ),
            # -------- Third Card --------
            ui.card(
                ui.card_header(ui.tags.span("Boosting", style="font-size:20px; font-weight:500;")
                ),

                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_slider("learning_rate", ui.HTML("learning_rate:<br>Size of improvement made by each tree"), 
                            min=0, 
                            max=len(learning_grid) - 1,
                            value=4,
                            step=1,
                            ticks=False,
                            animate=True
                        ),
                        ui.output_text("learning_rate_value"),
                        bg="#c6e3fc",
                        width="25%"
                    ),
                
                ui.output_plot("gb_learning_plot"),
                ui.output_ui("gb_learning_text"),
                full_screen=True
                ),
            ),
        ),

        # ---------- Page 3 ----------
        ui.nav_panel(
            "Comparative Analysis",
            ui.h4("Differences Between the Models"),

            # -------- First Line of Cards for Results --------

            ui.layout_columns( #first line 2 columns of cards

                ui.card(
                    ui.card_header(ui.tags.span("Model Performance Results", style="font-size:20px; font-weight:500;")
                    ),
                    ui.input_radio_buttons(
                    "model_results",
                    "Select a Model:",
                    ["Decision Tree", "Random Forest", "Boosting"],
                    inline=True   
                    ), 

                    ui.output_text_verbatim("model_performance")
                ),

                ui.card(
                    ui.card_header(ui.tags.span("Model Performance Comparison across Models", style="font-size:20px; font-weight:500;")
                    ),
                    ui.br(), #one line below
                    ui.br(),
                    ui.output_table("performance_comparison")
                )

            ),
            # -------- Second Line of Cards for Confusion Matrices --------

            ui.layout_columns( #second line 3 columns of cards

                ui.card( # Decision Tree Confusion Matrix
                    ui.card_header(ui.tags.span("Decision Tree Confusion Matrix", style="font-size:20px; font-weight:500;")
                    ),
                    ui.output_plot("dt_conf_matrix"),
                    full_screen=True
                ),

                ui.card( # Rf Confusion Matrix
                    ui.card_header(ui.tags.span("Random Forest Confusion Matrix", style="font-size:20px; font-weight:500;")
                    ),
                    ui.output_plot("rf_conf_matrix"),
                    full_screen=True
                ),
                ui.card( # Gb Confusion Matrix
                    ui.card_header(ui.tags.span("Boosting Model Confusion Matrix", style="font-size:20px; font-weight:500;")
                    ),
                    ui.output_plot("grd_conf_matrix"),
                    full_screen=True
                )
            ), 
            # -------- Third Card for ROC AUC --------
            ui.card(

                ui.card_header(ui.tags.span("ROC AUC Curve", style="font-size:20px; font-weight:500;")
                ),
                ui.input_selectize(
                    "roc_curve",
                    "Select ROC-AUC Plot:",
                    choices=[
                        "Decision Tree Model Plot",
                        "Random Forest Model Plot",
                        "Gradient Boosting Model Plot",
                        "Comparison Plot"
                    ],
                        multiple=False,
                        selected="Comparison Plot"
                ),
                ui.output_plot("roc_curve_plot"),
                style="height: 800px;", #making the card heigh larger since the table is collaped in defalt size
                full_screen=True
            ),
            # -------- Fourth Card for Tree Structures --------
            ui.card( 
                ui.card_header(ui.tags.span("Model Tree Structures", style="font-size:20px; font-weight:500;")
                               ),
                ui.input_selectize(
                    "tree_model",
                    "Select a Model: ",
                    choices=[
                        "Decision Tree Model",
                        "Random Forest Model",
                        "Gradient Boosting Model",
                    ],
                        multiple=False,
                        selected="Decision Tree Model"
                ),
                ui.output_plot("tree_plot", height="700px"),
                full_screen=True
            ),
            # -------- Fourth Card for Feature Importance --------
            ui.card(
                ui.card_header(ui.tags.span("Feature Importance by Model", style="font-size:20px; font-weight:500;")
                ),
                ui.input_radio_buttons(
                    "model_importance",
                    "Select a Model:",
                    choices= [
                        "Decision Tree Model",
                        "Random Forest Model",
                        "Gradient Boosting Model"],
                    selected= "Decision Tree Model",
                    inline=True   
                ), 

                ui.output_plot("feature_importance_plot"),
                full_screen=True
            
            ) 
        ),

        title="Diabetes Diagnosis Prediction with Tree-Based Mechine Learning Models",
        id="page",
    ),
    
)


def server(input, output, session):

    # data overview output
    @output
    @render.data_frame
    def df_head_table():
        return DataGrid(
        df_before_cleaning.head(10),
        width="100%",
        height="420px",
        filters=False,   
        editable=False
    )

    # hue plot output
    @output
    @render.plot
    def eda_hue_plot():

        chosen_variables = input.vars_hue_plot()
        default_vars = [
        "age",
        "height",
        "bmi",
        "cholesterol"
        ]
        if not chosen_variables or len(chosen_variables) < 2:
            vars_to_plot = default_vars
        else:
            vars_to_plot = chosen_variables

        hue_plot = sns.pairplot(
            df,
            vars = vars_to_plot,
            hue="diabetes"
        )

        return hue_plot.figure
    
    #distribution plot output
    @output
    @render.plot
    def eda_dist_plot():

        chosen_variable = input.var_dist_plot()

        default_variable = "age"

        if not chosen_variable:
            chosen_variable = default_variable

        dist_gender_mode = input.dist_gender_mode()

        kwargs = dict() #empthy dictionary to put the gender as row in the plot function using kwargs later

        if dist_gender_mode == "Off":
            kwargs.pop("row", None)
        elif dist_gender_mode == "On":
            kwargs["row"] = "gender"

        dist_plot = sns.displot(
            df, 
            x=chosen_variable,
            col="diabetes", 
            height=3, 
            facet_kws=dict(margin_titles=True),
            **kwargs
        )
        return dist_plot.figure
    
    #box plot output 
    @output
    @render.plot
    def eda_box_plot():

        chosen_variable = input.var_box_plot()

        default_variable = "age"

        if not chosen_variable:
            chosen_variable = default_variable

        box_gender_mode = input.box_gender_mode()

        kwargs = dict()

        if box_gender_mode == "Off":
            kwargs.pop("hue", None)
        elif box_gender_mode == "On":
            kwargs["hue"] = "gender"
        
        plt.figure(figsize=(10,7))

        box_plot = sns.boxplot(df,
            x="diabetes",
            y=chosen_variable,
            palette=["#1f77b4", "#ff7f0e"],
            **kwargs
        )    
        return box_plot.figure

    # heat map output
    @output
    @render.plot
    def heat_map_plot():
        chosen_variables = input.vars_heat_map()
        default_vars = [
        "age",
        "bmi",
        "cholesterol",
        "height"
        ]
        
        if not chosen_variables or len(chosen_variables) < 2:
            vars_to_plot = default_vars
        else:
            vars_to_plot = chosen_variables

        plt.figure(figsize=(10,8))

        corr = df[list(vars_to_plot)].corr() #calculating correlation

        heat_map = sns.heatmap(
            corr,
            annot=True, #showing numbers inside
            cmap="coolwarm",
            linewidths=0.5
        )
        return heat_map.figure

    # alpa value selected under slider
    @output
    @render.text
    def alpha_value():
        return f"Selected cpp_alpha value: {alpha_grid[int(input.alpha())]:.4f}"

    # alpha plot
    @output
    @render.plot
    def dt_alpha_plot():

        scores = treeCV.cv_results_["mean_test_score"]
        stds  = treeCV.cv_results_["std_test_score"]

        selected_alphas = int(input.alpha())

        plt.figure(figsize=(8, 5))

        dt_alpha = sns.lineplot(x=alpha_grid, y=scores, marker="o")
        plt.errorbar(alpha_grid, scores, yerr=stds, fmt="none", capsize=3)

        plt.scatter(alpha_grid[selected_alphas], scores[selected_alphas], s=150)

        plt.xscale("log")
        plt.xlabel("ccp_alpha (log scale)")
        plt.ylabel("10-fold CV Accuracy")
        plt.title("Decision Tree Performance Changes with Alpha")
        plt.tight_layout()

        return dt_alpha.figure

    # alpha cross validatidation explanation text
    @output
    @render.ui        #using html rather than text to be able to adjust text better
    def dt_alpha_text():

        scores  = treeCV.cv_results_["mean_test_score"]
        stds   = treeCV.cv_results_["std_test_score"]

        selected_alpha = int(input.alpha())
        best_alpha = treeCV.best_index_
        
        # Converting scores and stds to percentages
        accuracy = scores[selected_alpha] * 100
        std_dev  = stds[selected_alpha] * 100

        text = ""

        text += "<span style='font-weight: normal;'>"

        text += "During model tuning, the model achieved an average accuracy of %.1f%% " % accuracy

        text += "with a standard deviation of %.1f%%. " % std_dev

        text += "This indicates consistent model performance across the validation tests."

        text += "</span>"

        # Checking if best parameter selected
        if selected_alpha == best_alpha:
            text += "<b>This complexity level gives the best model performance.</b>"
        else:
            text += "<b>Other complexity levels give better model performance.</b>"

        return ui.HTML(text)

    # max_features value selected under slider
    @output
    @render.text
    def max_features_value():
        return f"Selected number of features: {input.max_features()}"

    #max_features plot
    
    @output
    @render.plot
    def rf_features_plot():

        scores = rfCV_feature.cv_results_["mean_test_score"]
        stds  = rfCV_feature.cv_results_["std_test_score"]

        selected_feature = input.max_features()

        index= feature_grid.index(selected_feature)

        plt.figure(figsize=(8, 5))

        rf_feature = sns.lineplot(x=feature_grid, y=scores, marker="o")
        plt.errorbar(feature_grid, scores, yerr=stds, fmt="none", capsize=3)

        plt.scatter(feature_grid[index], scores[index], s=150)

        plt.xlabel("Maximum Number of Features")
        plt.ylabel("10-fold CV Accuracy")
        plt.title("Random Forest Performance Changes with Number of Features")
        plt.tight_layout()

        return rf_feature.figure
    
   
    #max_features cross validatidation explanation text
    @output
    @render.ui
    def rf_features_text():
        scores  = rfCV_feature.cv_results_["mean_test_score"]
        stds   = rfCV_feature.cv_results_["std_test_score"]

        selected_feature = input.max_features()
        best_index = rfCV_feature.best_index_
        index= feature_grid.index(selected_feature)

        accuracy = scores[index] * 100
        std_dev  = stds[index] * 100

        text = ""

        text += "<span style='font-weight: normal;'>"

        text += "The model achieved an average accuracy of %.1f%% " % accuracy
        text += "with a standard deviation of %.1f%%. " % std_dev

        text += "This indicates consistent model performance across the validation tests."

        text += "</span>"

        if index == best_index:
            text += "<b>This number of features gives the best model performance.</b>"
        else:
            text += "<b>Other values of the number of features give better model performance.</b>"

        return ui.HTML(text)
    
    #learning_rate value selected under slider
    @output
    @render.text
    def learning_rate_value():
        return f"Selected learning rate value: {learning_grid[int(input.learning_rate())]:.4f}"

    #learning rate plot
    @output
    @render.plot
    def gb_learning_plot():

        scores = grdCV.cv_results_["mean_test_score"]
        stds  = grdCV.cv_results_["std_test_score"]

        selected_learning = int(input.learning_rate())

        plt.figure(figsize=(8, 5))

        gb_learning = sns.lineplot(x=learning_grid, y=scores, marker="o")
        plt.errorbar(learning_grid, scores, yerr=stds, fmt="none", capsize=3)

        plt.scatter(learning_grid[selected_learning], scores[selected_learning], s=150)

        plt.xscale("log")
        plt.xlabel("Learning rate (log scale)")
        plt.ylabel("10-fold CV Accuracy")
        plt.title("Gradient Boosting Tree Performance Changes with Learning Rate")
        plt.tight_layout()

        return gb_learning.figure
    
    #learning_rate cross validatidation explanation text
    @output
    @render.ui        
    def gb_learning_text():

        scores  = grdCV.cv_results_["mean_test_score"]
        stds   = grdCV.cv_results_["std_test_score"]

        selected_learning = int(input.learning_rate())
        best_index = grdCV.best_index_

        accuracy = scores[selected_learning] * 100
        std_dev  = stds[selected_learning] * 100


        text = ""

        text += "<span style='font-weight: normal;'>" #these are normal not bold

        text += "The model achieved an average accuracy of %.1f%% " % accuracy
        text += "with a standard deviation of %.1f%%. " % std_dev
        text += "This indicates consistent model performance across the validation tests. "

        text += "</span>" #these are normal not bold

        if selected_learning == best_index: #if best rate
            text += "<b>This learning rate gives the best model performance.</b>" #bold
        else:
            text += "<b>Other learning rate values give better model performance.</b>" #bold

        return ui.HTML(text)


    # classification report outputs by model 
    @output
    @render.text
    def model_performance():

        model = input.model_results()

        if model == "Decision Tree":
            y_pred = y_pred_dt

        elif model == "Random Forest":
            y_pred = y_pred_rf

        else:  # Boosting
            y_pred = y_pred_grd


        report = classification_report(
            y_test,
            y_pred,
            target_names=["No diabetes","Diabetes"]
        )

        return f"{model} Model Results:\n\n{report}"
    
    # performance comparison table
    @output
    @render.table
    def performance_comparison():
        table = pd.DataFrame({

            "Model": [
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting"
            ],

            "Accuracy": [
                accuracy_score(y_test, y_pred_dt),
                accuracy_score(y_test, y_pred_rf),
                accuracy_score(y_test, y_pred_grd),
            ],

            "Precision": [
                precision_score(y_test, y_pred_dt),
                precision_score(y_test, y_pred_rf),
                precision_score(y_test, y_pred_grd)
            ],

            "Sensitivity (Recall)": [
                recall_score(y_test, y_pred_dt),
                recall_score(y_test, y_pred_rf),
                recall_score(y_test, y_pred_grd)
            ],

            "F1-score": [
                f1_score(y_test, y_pred_dt),
                f1_score(y_test, y_pred_rf),
                f1_score(y_test, y_pred_grd)
            ]
        })  

        return table.round(3).reset_index(drop=True)

    # dt confusion matrix
    @output
    @render.plot
    def dt_conf_matrix():
        dt_conf = confusion_matrix(y_test, y_pred_dt)
        fig = ConfusionMatrixDisplay(dt_conf, display_labels = ["No diabetes", "Diabetes"])
        fig.plot()
    
        return plt.gcf()
    
     # rf confusion matrix
    @output
    @render.plot
    def rf_conf_matrix():
        rf_conf = confusion_matrix(y_test, y_pred_rf)
        fig = ConfusionMatrixDisplay(rf_conf, display_labels = ["No diabetes", "Diabetes"])
        fig.plot()
    
        return plt.gcf()
     # grd confusion matrix
    @output
    @render.plot
    def grd_conf_matrix():
        grd_conf = confusion_matrix(y_test, y_pred_grd)
        fig = ConfusionMatrixDisplay(grd_conf, display_labels = ["No diabetes", "Diabetes"])
        fig.plot()
    
        return plt.gcf()    

    #roc curve plot individual and comparison
    @output
    @render.plot
    def roc_curve_plot():
        selected_plot = input.roc_curve()

        plt.figure(figsize=(10,10))

        # Decision Tree
        if selected_plot == "Decision Tree Model Plot":
            plt.title("Decision Tree ROC Curve")
            plt.plot(fpr, tpr, 'b',label=f"DT (AUC={auc:.2f})")

        # Random Forest
        elif selected_plot == "Random Forest Model Plot":
            plt.title("Random Forest ROC Curve")
            plt.plot(fpr_rf, tpr_rf, 'g',label=f"RF (AUC={auc_rf:.2f})")

        # Gradient Boosting
        elif selected_plot == "Gradient Boosting Model Plot":
            plt.title("Gradient Boosting ROC Curve")
            plt.plot(fpr_grd, tpr_grd,'c', label=f"GB (AUC={auc_grd:.2f})")

        # Comparison Plot
        else:
            plt.title("ROC Curve Comparison of Tree-Based Models")
            plt.plot(fpr, tpr, 'b', label=f"DT (AUC = {auc:.2f})")
            plt.plot(fpr_rf, tpr_rf, 'g', label=f"RF (AUC = {auc_rf:.2f})")
            plt.plot(fpr_grd, tpr_grd, 'c', label=f"GB (AUC = {auc_grd:.2f})")

        # Formatting
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        return plt.gcf()

    #tree visualisations
    @output
    @render.plot
    def tree_plot():
        selected_model = input.tree_model()
        fn = X_train_resampled.columns
        cn = ["No diabetes", "Diabetes"]


        if selected_model == "Decision Tree Model":
            fig, ax = subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=150)
            tree.plot_tree(
                dt_model,
                feature_names=fn,
                class_names=cn,
                filled=True,
                rounded=True,
                max_depth=3,
                fontsize=8,
                ax=ax
            )
            ax.set_title("Decision Tree Structure", fontsize=10)

        elif selected_model == "Random Forest Model":  
            fig, ax = subplots(nrows=1, ncols=2, figsize=(30, 15), dpi=150) #creating colums to sit the trees
            for i in range(0,2): # showing 2 example trees
                plot_tree(
                    rf_model.estimators_[i], 
                    feature_names=fn,
                    class_names=cn,
                    filled=True,
                    rounded=True,
                    max_depth=3,
                    fontsize=3,
                    ax=ax[i]
                )
                ax[i].set_title(f"Example Tree: {i+1}", fontsize = 8)
            fig.suptitle("Structure of Example Random Forest Trees", fontsize=10) # multiple axes

        else:  # "Gradient Boosting"
            fig, ax = subplots(nrows=1, ncols=2, figsize=(16, 8), dpi=150)
            for i in range(2): # showing 2 example trees
                plot_tree(
                    grd_model.estimators_[i,0], 
                    feature_names=fn,
                    filled=True,
                    rounded=True,
                    max_depth=3,
                    fontsize=5,
                    ax=ax[i]
                )
                ax[i].set_title(f"Example Tree: {i+1}", fontsize = 8)
            fig.suptitle("Structure of Example Gradient Boosting Trees", fontsize=10)

        fig.tight_layout()
        return fig

    #feature importance plot
    @output
    @render.plot
    def feature_importance_plot():
        selected_model = input.model_importance()

        if selected_model == "Decision Tree Model":
            model_importances = dt_model.feature_importances_
        elif selected_model == "Random Forest Model":
            model_importances = rf_model.feature_importances_
        else:
            model_importances = grd_model.feature_importances_
    
        importances = pd.Series(
            model_importances,
            index=X.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(7,5))

        importances.head(10).sort_values().plot(kind="barh")

        plt.xlabel("Mean decrease in impurity")
        plt.ylabel("Features")
        plt.title(f"{selected_model} Feature Importance", fontsize=12)

        plt.tight_layout()
        return plt.gcf()






app = App(app_ui, server)