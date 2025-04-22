# app.R - Optimized Regression Model Comparison Shiny App

# Load required libraries
library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(rpart)
library(ggplot2)
library(DT)
library(plotly)
library(shinyWidgets)

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "SmartML"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Upload Data", tabName = "upload", icon = icon("upload")),
      menuItem("Model Training", tabName = "training", icon = icon("cogs")),
      menuItem("Model Evaluation", tabName = "evaluation", icon = icon("chart-line")),
      menuItem("Feature Importance", tabName = "importance", icon = icon("list-ol")),
      menuItem("Predictions", tabName = "predictions", icon = icon("chart-area")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      # Data Upload Tab
      tabItem(tabName = "upload",
              fluidRow(
                box(
                  title = "Upload Dataset", status = "primary", solidHeader = TRUE, width = 12,
                  fileInput("file", "Choose CSV File",
                            accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")),
                  checkboxInput("header", "Header", TRUE),
                  radioButtons("sep", "Separator",
                               choices = c(Comma = ",", Semicolon = ";", Tab = "\t"),
                               selected = ","),
                  checkboxInput("useSample", "Use Sample Data for Quick Exploration (10% of data)", FALSE),
                  sliderInput("sampleSize", "Sample Size (%):", min = 5, max = 50, value = 10, step = 5)
                )
              ),
              fluidRow(
                box(
                  title = "Dataset Preview", status = "info", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("dataTable"),
                  uiOutput("targetSelector")
                )
              )
      ),
      
      # Model Training Tab
      tabItem(tabName = "training",
              fluidRow(
                box(
                  title = "Training Settings", status = "primary", solidHeader = TRUE, width = 6,
                  sliderInput("trainSplit", "Training/Test Split Ratio:", min = 0.5, max = 0.9, value = 0.8, step = 0.05),
                  sliderInput("cvFolds", "Number of CV Folds:", min = 2, max = 10, value = 3, step = 1),
                  selectInput("selectedModels", "Select Models to Train:", 
                              choices = list("Decision Tree" = "dt", "Random Forest" = "rf", 
                                             "Support Vector Machine" = "svm", "Gradient Boosting" = "gb"),
                              multiple = TRUE, selected = c("dt")),
                  radioButtons("trainMode", "Training Mode:",
                               choices = c("Train All Selected Models" = "all", "Train Single Model" = "single"),
                               selected = "single"),
                  conditionalPanel(
                    condition = "input.trainMode == 'single'",
                    selectInput("singleModel", "Select Single Model:", 
                                choices = list("Decision Tree" = "dt", "Random Forest" = "rf", 
                                               "Support Vector Machine" = "svm", "Gradient Boosting" = "gb"),
                                selected = "dt")
                  ),
                  numericInput("seed", "Random Seed:", value = 123, min = 1, max = 9999),
                  actionButton("trainButton", "Train Models", class = "btn-success")
                ),
                box(
                  title = "Hyperparameter Settings", status = "warning", solidHeader = TRUE, width = 6,
                  conditionalPanel(
                    condition = "input.selectedModels.includes('dt') || input.singleModel == 'dt'",
                    sliderInput("dtCP", "Decision Tree Complexity Parameter:", min = 0.001, max = 0.1, value = 0.01, step = 0.001)
                  ),
                  conditionalPanel(
                    condition = "input.selectedModels.includes('rf') || input.singleModel == 'rf'",
                    sliderInput("rfMtry", "Random Forest mtry:", min = 2, max = 10, value = 3, step = 1)
                  ),
                  conditionalPanel(
                    condition = "input.selectedModels.includes('svm') || input.singleModel == 'svm'",
                    sliderInput("svmSigma", "SVM Sigma:", min = 0.01, max = 0.1, value = 0.05, step = 0.01),
                    sliderInput("svmC", "SVM Cost:", min = 0.1, max = 5, value = 1, step = 0.1)
                  ),
                  conditionalPanel(
                    condition = "input.selectedModels.includes('gb') || input.singleModel == 'gb'",
                    sliderInput("gbTrees", "GB n.trees:", min = 100, max = 300, value = 200, step = 50),
                    sliderInput("gbDepth", "GB interaction.depth:", min = 1, max = 5, value = 3, step = 1),
                    sliderInput("gbShrinkage", "GB shrinkage:", min = 0.01, max = 0.1, value = 0.05, step = 0.01)
                  )
                )
              ),
              fluidRow(
                box(
                  title = "Training Progress", status = "info", solidHeader = TRUE, width = 12,
                  verbatimTextOutput("trainingLog"),
                  uiOutput("trainingStatus"),
                  shinyWidgets::progressBar(id = "trainProgress", value = 0, total = 100, display_pct = TRUE)
                )
              )
      ),
      
      # Model Evaluation Tab
      tabItem(tabName = "evaluation",
              fluidRow(
                box(
                  title = "Model Performance Metrics", status = "primary", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("modelMetrics")
                )
              ),
              fluidRow(
                box(
                  title = "Cross-Validation Results", status = "info", solidHeader = TRUE, width = 6,
                  plotlyOutput("cvResults")
                ),
                box(
                  title = "Test Set Performance", status = "warning", solidHeader = TRUE, width = 6,
                  plotlyOutput("testResults")
                )
              )
      ),
      
      # Feature Importance Tab
      tabItem(tabName = "importance",
              fluidRow(
                box(
                  title = "Feature Importance", status = "primary", solidHeader = TRUE, width = 12,
                  selectInput("importanceModel", "Select Model:", 
                              choices = list("Decision Tree" = "dt", "Random Forest" = "rf", 
                                             "Support Vector Machine" = "svm", "Gradient Boosting" = "gb"),
                              selected = "rf"),
                  plotlyOutput("featureImportance")
                )
              ),
              fluidRow(
                box(
                  title = "Recursive Feature Elimination", status = "info", solidHeader = TRUE, width = 12,
                  checkboxInput("runRFE", "Run Recursive Feature Elimination (can be time-consuming)", FALSE),
                  plotlyOutput("rfePlot"),
                  verbatimTextOutput("selectedFeatures")
                )
              )
      ),
      
      # Predictions Tab
      tabItem(tabName = "predictions",
              fluidRow(
                box(
                  title = "Actual vs Predicted", status = "primary", solidHeader = TRUE, width = 6,
                  selectInput("predModel", "Select Model for Predictions:", 
                              choices = list("Decision Tree" = "DT", "Random Forest" = "RF", 
                                             "Support Vector Machine" = "SVM", "Gradient Boosting" = "GB")),
                  plotlyOutput("actualVsPredicted")
                ),
                box(
                  title = "Residual Analysis", status = "warning", solidHeader = TRUE, width = 6,
                  plotlyOutput("residualPlot")
                )
              ),
              fluidRow(
                box(
                  title = "Prediction Results", status = "info", solidHeader = TRUE, width = 12,
                  DT::dataTableOutput("predictionResults")
                )
              )
      ),
      
      # About Tab
      tabItem(tabName = "about",
              fluidRow(
                box(
                  title = "About This App", status = "primary", solidHeader = TRUE, width = 12,
                  h3("SmartML: Train, Evaluate, Optimize, Tune, Select"),
                  p("SmartML is an interactive application designed to tackle problems with continuous target variables using advanced machine learning algorithms. It features tools for model training, evaluation, and optimization, including hyperparameter tuning, feature importance analysis, and feature selection. With these capabilities, SmartML empowers users to effectively address  challenges and enhance their predictive modeling outcomes.This app implements the following  algorithms:"),
                  tags$ul(
                    tags$li("Decision Tree (DT): A tree-based model that splits data based on feature values to predict continuous outcomes."),
                    tags$li("Random Forest (RF): An ensemble method using multiple decision trees to improve accuracy and reduce overfitting."),
                    tags$li("Support Vector Machine (SVM): A kernel-based method that finds the optimal hyperplane for regression tasks."),
                    tags$li("Gradient Boosting (GB): An ensemble technique that builds trees sequentially to minimize prediction errors."),
                  ),
                  h3("Contact"),
                  p("For inquiries or support, please contact:"),
                  tags$ul(
                    tags$li(a(href = "mailto:imstat09@gmail.com", "Email: imstat09@gmail.com")),
                    tags$li(a(href = "https://www.youtube.com/@Iqbalstat", "YouTube Channel: Iqbalstat"))
                  ),
                  h3("Developed By"),
                  p("M. Iqbal Jeelani (SKUAST-Kashmir)")
                )
              )
      )
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  # Reactive values for storing data and models
  values <- reactiveValues(
    data = NULL,
    raw_data = NULL,
    train_data = NULL,
    test_data = NULL,
    models = list(),
    results = NULL,
    rfe_results = NULL,
    predictions = NULL,
    training_complete = FALSE
  )
  
  # Read data when file is uploaded
  observeEvent(input$file, {
    req(input$file)
    values$raw_data <- read.csv(input$file$datapath, header = input$header, sep = input$sep)
    values$data <- values$raw_data
    values$training_complete <- FALSE
    values$models <- list()
  })
  
  # Handle sample data
  observe({
    req(values$raw_data)
    if (input$useSample) {
      set.seed(input$seed)
      sample_size <- floor(nrow(values$raw_data) * (input$sampleSize / 100))
      values$data <- values$raw_data[sample(nrow(values$raw_data), sample_size), ]
    } else {
      values$data <- values$raw_data
    }
  })
  
  # Generate data preview table
  output$dataTable <- DT::renderDataTable({
    req(values$data)
    DT::datatable(head(values$data, 100), options = list(scrollX = TRUE))
  })
  
  # Generate target variable selector
  output$targetSelector <- renderUI({
    req(values$data)
    selectInput("targetVar", "Select Target Variable:", 
                choices = names(values$data), 
                selected = names(values$data)[ncol(values$data)])
  })
  
  # Train models when button is clicked
  observeEvent(input$trainButton, {
    req(values$data, input$targetVar)
    
    # Set seed for reproducibility
    set.seed(input$seed)
    
    # Create a progress object
    progress <- shiny::Progress$new()
    progress$set(message = "Initializing training", value = 0)
    on.exit(progress$close())
    
    # Update the training log
    output$trainingLog <- renderText({
      "Starting model training...\n"
    })
    
    # Clean the data - Remove NA values
    data <- values$data
    data <- na.omit(data)
    
    # Ensure target variable is numeric
    if (!is.numeric(data[[input$targetVar]])) {
      output$trainingLog <- renderText({
        "Error: Target variable must be numeric for regression models!"
      })
      return()
    }
    
    # Rename target column for easier reference
    names(data)[names(data) == input$targetVar] <- "target"
    
    # Split data into training and testing sets
    trainIndex <- createDataPartition(data$target, p = input$trainSplit, list = FALSE)
    values$train_data <- data[trainIndex, ]
    values$test_data <- data[-trainIndex, ]
    
    output$trainingLog <- renderText({
      paste0("Data split into ", nrow(values$train_data), " training samples and ", 
             nrow(values$test_data), " test samples.\n",
             "Starting model training with ", input$cvFolds, " fold cross-validation...\n")
    })
    
    # Define evaluation metrics (RMSE, MAE, MAPE)
    custom_metrics <- function(data, lev = NULL, model = NULL) {
      rmse <- sqrt(mean((data$obs - data$pred)^2))
      mae <- mean(abs(data$obs - data$pred))
      mape <- mean(abs((data$obs - data$pred) / ifelse(data$obs == 0, 1e-10, data$obs))) * 100
      c(RMSE = rmse, MAE = mae, MAPE = mape)
    }
    
    # Train control for cross-validation
    ctrl <- trainControl(method = "cv", number = input$cvFolds, 
                         summaryFunction = custom_metrics, savePredictions = TRUE)
    
    values$models <- list()
    models_to_train <- if (input$trainMode == "single") input$singleModel else input$selectedModels
    model_count <- length(models_to_train)
    progress_step <- 100 / model_count
    
    # Train selected models
    for (i in seq_along(models_to_train)) {
      model_type <- models_to_train[i]
      progress$set(value = progress_step * (i - 1), 
                   detail = paste("Training", model_type, "model"))
      
      shinyWidgets::updateProgressBar(session = session, id = "trainProgress", 
                                      value = progress_step * (i - 1))
      
      output$trainingLog <- renderText({
        paste0("Training ", model_type, " model (", i, " of ", model_count, ")...\n")
      })
      
      if (model_type == "dt") {
        dt_grid <- expand.grid(cp = c(input$dtCP, input$dtCP * 2))
        values$models$DT <- tryCatch({
          train(target ~ ., data = values$train_data, method = "rpart",
                trControl = ctrl, tuneGrid = dt_grid, metric = "RMSE")
        }, error = function(e) {
          output$trainingLog <- renderText({
            paste0("Error training Decision Tree model: ", e$message, "\n")
          })
          return(NULL)
        })
      } else if (model_type == "rf") {
        rf_grid <- expand.grid(mtry = c(input$rfMtry, input$rfMtry + 3))
        values$models$RF <- tryCatch({
          train(target ~ ., data = values$train_data, method = "rf",
                trControl = ctrl, tuneGrid = rf_grid, metric = "RMSE", importance = TRUE)
        }, error = function(e) {
          output$trainingLog <- renderText({
            paste0("Error training Random Forest model: ", e$message, "\n")
          })
          return(NULL)
        })
      } else if (model_type == "svm") {
        svm_grid <- expand.grid(sigma = c(input$svmSigma, input$svmSigma * 2), 
                                C = c(input$svmC, input$svmC * 2))
        values$models$SVM <- tryCatch({
          train(target ~ ., data = values$train_data, method = "svmRadial",
                trControl = ctrl, tuneGrid = svm_grid, metric = "RMSE")
        }, error = function(e) {
          output$trainingLog <- renderText({
            paste0("Error training SVM model: ", e$message, "\n")
          })
          return(NULL)
        })
      } else if (model_type == "gb") {
        gb_grid <- expand.grid(n.trees = c(input$gbTrees, input$gbTrees + 50),
                               interaction.depth = c(input$gbDepth, input$gbDepth + 1),
                               shrinkage = c(input$gbShrinkage, input$gbShrinkage * 2),
                               n.minobsinnode = 10)
        values$models$GB <- tryCatch({
          train(target ~ ., data = values$train_data, method = "gbm",
                trControl = ctrl, tuneGrid = gb_grid, metric = "RMSE", verbose = FALSE)
        }, error = function(e) {
          output$trainingLog <- renderText({
            paste0("Error training Gradient Boosting model: ", e$message, "\n")
          })
          return(NULL)
        })
      }
      
      progress$set(value = progress_step * i)
      shinyWidgets::updateProgressBar(session = session, id = "trainProgress", 
                                      value = progress_step * i)
    }
    
    # Create model evaluation results
    evaluate_model <- function(model, test_data) {
      if (is.null(model)) return(c(RMSE = NA, MAE = NA, MAPE = NA))
      
      pred <- predict(model, test_data)
      obs <- test_data$target
      rmse <- sqrt(mean((obs - pred)^2))
      mae <- mean(abs(obs - pred))
      mape <- mean(abs((obs - pred) / ifelse(obs == 0, 1e-10, obs))) * 100
      return(c(RMSE = rmse, MAE = mae, MAPE = mape))
    }
    
    # Evaluate all models
    values$results <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), MAPE = numeric())
    for (name in names(values$models)) {
      if (!is.null(values$models[[name]])) {
        metrics <- evaluate_model(values$models[[name]], values$test_data)
        values$results <- rbind(values$results, data.frame(Model = name, RMSE = metrics["RMSE"], 
                                                          MAE = metrics["MAE"], MAPE = metrics["MAPE"]))
      }
    }
    
    # Generate predictions for each model
    values$test_data_with_pred <- values$test_data
    for (name in names(values$models)) {
      if (!is.null(values$models[[name]])) {
        values$test_data_with_pred[[paste0(name, "_pred")]] <- predict(values$models[[name]], values$test_data)
      }
    }
    
    # Update training status
    values$training_complete <- TRUE
    
    output$trainingStatus <- renderUI({
      if (values$training_complete) {
        div(
          p("All models have been trained successfully!", style = "color: green; font-weight: bold;"),
          p("You can now explore the model evaluation, feature importance, and predictions tabs.")
        )
      }
    })
    
    output$trainingLog <- renderText({
      paste0("All models trained successfully!\n",
             "Summary of trained models:\n",
             paste(names(values$models), collapse = ", "), "\n\n",
             "Proceed to the Model Evaluation tab to compare model performance.")
    })
  })
  
  # Generate model metrics table
  output$modelMetrics <- DT::renderDataTable({
    req(values$results)
    DT::datatable(values$results, options = list(dom = 't')) %>%
      formatRound(columns = c("RMSE", "MAE", "MAPE"), digits = 4)
  })
  
  # Generate CV results plot
  output$cvResults <- renderPlotly({
    req(values$models)
    
    cv_data <- data.frame()
    
    for (name in names(values$models)) {
      if (!is.null(values$models[[name]])) {
        model_cv <- values$models[[name]]$results
        # Select only common metric columns and add Model name
        common_cols <- intersect(colnames(model_cv), c("RMSE", "MAE", "MAPE"))
        if (length(common_cols) > 0) {
          model_cv <- model_cv[, common_cols, drop = FALSE]
          model_cv$Model <- name
          cv_data <- rbind(cv_data, model_cv)
        }
      }
    }
    
    if (nrow(cv_data) > 0) {
      p <- ggplot(cv_data, aes(x = Model, y = RMSE, fill = Model)) +
        geom_boxplot() +
        labs(title = "Cross-Validation RMSE by Model", x = "Model", y = "RMSE") +
        theme_minimal()
      
      ggplotly(p)
    }
  })
  
  # Generate test results plot
  output$testResults <- renderPlotly({
    req(values$results)
    
    results_long <- tidyr::pivot_longer(values$results, cols = c("RMSE", "MAE", "MAPE"), 
                                        names_to = "Metric", values_to = "Value")
    
    p <- ggplot(results_long, aes(x = Model, y = Value, fill = Model)) +
      geom_bar(stat = "identity") +
      facet_wrap(~Metric, scales = "free_y") +
      labs(title = "Test Set Performance by Model", x = "Model", y = "Value") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggplotly(p)
  })
  
  # Generate feature importance plot
  output$featureImportance <- renderPlotly({
    req(values$models, input$importanceModel, values$train_data, values$test_data)
    
    if (input$importanceModel == "dt" && !is.null(values$models$DT)) {
      dt_imp <- varImp(values$models$DT)$importance
      dt_imp$Feature <- rownames(dt_imp)
      p <- ggplot(dt_imp, aes(x = reorder(Feature, Overall), y = Overall)) +
        geom_bar(stat = "identity", fill = "brown") +
        coord_flip() +
        labs(title = "Decision Tree Feature Importance", x = "Feature", y = "Importance") +
        theme_minimal()
    } else if (input$importanceModel == "rf" && !is.null(values$models$RF)) {
      rf_imp <- varImp(values$models$RF)$importance
      rf_imp$Feature <- rownames(rf_imp)
      p <- ggplot(rf_imp, aes(x = reorder(Feature, Overall), y = Overall)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(title = "Random Forest Feature Importance", x = "Feature", y = "Importance") +
        theme_minimal()
    } else if (input$importanceModel == "svm" && !is.null(values$models$SVM)) {
      # Permutation importance for SVM using the train object
      set.seed(input$seed)
      pred_base <- predict(values$models$SVM, values$test_data)
      rmse_base <- sqrt(mean((values$test_data$target - pred_base)^2))
      svm_imp <- data.frame(Overall = numeric(), Feature = character())
      predictors <- names(values$test_data)[-which(names(values$test_data) == "target")]
      for (feat in predictors) {
        temp_data <- values$test_data
        temp_data[[feat]] <- sample(temp_data[[feat]])
        pred_perm <- predict(values$models$SVM, temp_data)
        rmse_perm <- sqrt(mean((values$test_data$target - pred_perm)^2))
        imp <- (rmse_perm - rmse_base) / rmse_base * 100
        svm_imp <- rbind(svm_imp, data.frame(Overall = imp, Feature = feat))
      }
      p <- ggplot(svm_imp, aes(x = reorder(Feature, Overall), y = Overall)) +
        geom_bar(stat = "identity", fill = "purple") +
        coord_flip() +
        labs(title = "SVM Feature Importance (Permutation)", x = "Feature", y = "Importance (%)") +
        theme_minimal()
    } else if (input$importanceModel == "gb" && !is.null(values$models$GB)) {
      gb_imp <- summary(values$models$GB$finalModel, plot = FALSE)
      p <- ggplot(gb_imp, aes(x = reorder(var, rel.inf), y = rel.inf)) +
        geom_bar(stat = "identity", fill = "darkgreen") +
        coord_flip() +
        labs(title = "Gradient Boosting Feature Importance", x = "Feature", y = "Relative Influence") +
        theme_minimal()
    }
    
    if (exists("p")) ggplotly(p)
  })
  
  # Perform RFE and generate plot
  observe({
    req(values$train_data, input$runRFE)
    
    if (input$runRFE && is.null(values$rfe_results)) {
      output$selectedFeatures <- renderText({
        "Running Recursive Feature Elimination... (this may take a while)"
      })
      
      withProgress(message = 'Running RFE...', value = 0, {
        tryCatch({
          rfe_ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = min(input$cvFolds, 3))
          max_features <- min(ncol(values$train_data) - 1, 15)
          
          values$rfe_results <- rfe(values$train_data[, -which(names(values$train_data) == "target")], 
                                    values$train_data$target,
                                    sizes = c(1:max_features), 
                                    rfeControl = rfe_ctrl, 
                                    metric = "RMSE")
        }, error = function(e) {
          output$selectedFeatures <- renderText({
            paste("Error in RFE:", e$message)
          })
          values$rfe_results <- NULL
        })
      })
    }
    
    output$rfePlot <- renderPlotly({
      req(values$rfe_results)
      
      rfe_data <- data.frame(Variables = values$rfe_results$results$Variables, 
                             RMSE = values$rfe_results$results$RMSE)
      
      p <- ggplot(rfe_data, aes(x = Variables, y = RMSE)) +
        geom_line(color = "blue") +
        geom_point(color = "red") +
        labs(title = "RFE: RMSE vs Number of Features", x = "Number of Features", y = "RMSE") +
        theme_minimal()
      
      ggplotly(p)
    })
    
    output$selectedFeatures <- renderText({
      req(values$rfe_results)
      
      paste("Optimal Number of Features:", length(values$rfe_results$optVariables), "\n",
            "Selected Features:", paste(values$rfe_results$optVariables, collapse = ", "))
    })
  })
  
  # Generate actual vs predicted plot
  output$actualVsPredicted <- renderPlotly({
    req(values$test_data_with_pred, input$predModel)
    
    pred_col <- paste0(input$predModel, "_pred")
    
    if (!pred_col %in% names(values$test_data_with_pred)) {
      return(NULL)
    }
    
    p <- ggplot(values$test_data_with_pred, aes_string(x = "target", y = pred_col)) +
      geom_point(color = "blue", alpha = 0.5) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      labs(title = paste0(input$predModel, ": Actual vs Predicted"), x = "Actual", y = "Predicted") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # Generate residual plot
  output$residualPlot <- renderPlotly({
    req(values$test_data_with_pred, input$predModel)
    
    pred_col <- paste0(input$predModel, "_pred")
    
    if (!pred_col %in% names(values$test_data_with_pred)) {
      return(NULL)
    }
    
    residuals <- values$test_data_with_pred$target - values$test_data_with_pred[[pred_col]]
    
    p <- ggplot(data.frame(Predicted = values$test_data_with_pred[[pred_col]], Residuals = residuals), 
                aes(x = Predicted, y = Residuals)) +
      geom_point(color = "purple", alpha = 0.5) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      labs(title = paste0(input$predModel, ": Residual Plot"), x = "Predicted", y = "Residuals") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # Generate prediction results table
  output$predictionResults <- DT::renderDataTable({
    req(values$test_data_with_pred)
    
    pred_cols <- grep("_pred$", names(values$test_data_with_pred), value = TRUE)
    
    if (length(pred_cols) == 0) {
      return(NULL)
    }
    
    results_table <- values$test_data_with_pred[, c("target", pred_cols)]
    
    DT::datatable(results_table, options = list(scrollX = TRUE)) %>%
      formatRound(columns = c("target", pred_cols), digits = 4)
  })
}

# Run the application
shinyApp(ui = ui, server = server)
