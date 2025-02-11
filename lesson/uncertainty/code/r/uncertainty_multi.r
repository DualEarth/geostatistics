# Load necessary libraries
library(ggplot2)

# Set seed for reproducibility
set.seed(123)

### Step 1: Generate Synthetic Data with Known Uncertainty ###
n <- 100  # Number of data points

# Generate predictor variables (X1, X2) from log-normal distributions
X1 <- rlnorm(n, meanlog = 1, sdlog = 0.2)  
X2 <- rlnorm(n, meanlog = 1.5, sdlog = 0.3)  

# True model parameters (with known uncertainty)
beta0_true <- rnorm(1, mean = 2, sd = 0.5)  
beta1_true <- rnorm(1, mean = 1.2, sd = 0.2)  
beta2_true <- rnorm(1, mean = -0.8, sd = 0.15)  

# Generate response variable Y with log-normal noise
epsilon <- rlnorm(n, meanlog = 0, sdlog = 0.1)  
Y_true <- exp(beta0_true + beta1_true * log(X1) + beta2_true * log(X2)) * epsilon

# Store in a data frame
data <- data.frame(X1, X2, Y_true)

### Step 2: Fit a Deterministic Linear Regression Model ###
lm_model <- lm(log(Y_true) ~ log(X1) + log(X2), data = data)  
data$Y_pred <- exp(predict(lm_model))  

# Compute R² for deterministic model
R2_deterministic <- summary(lm_model)$r.squared

### Step 3: Stochastic Model - Monte Carlo Ensemble ###
n_sim <- 1000  # Number of stochastic realizations

# Storage for stochastic predictions
Y_simulated <- matrix(NA, nrow = n, ncol = n_sim)

for (i in 1:n_sim) {
  # Sample assumed uncertainty in inputs
  X1_sim <- rlnorm(n, meanlog = log(X1), sdlog = 0.1)  
  X2_sim <- rlnorm(n, meanlog = log(X2), sdlog = 0.15)
  
  # Sample uncertainty in model parameters
  beta0_sim <- rnorm(1, mean = coef(lm_model)[1], sd = 0.3)
  beta1_sim <- rnorm(1, mean = coef(lm_model)[2], sd = 0.1)
  beta2_sim <- rnorm(1, mean = coef(lm_model)[3], sd = 0.1)
  
  # Simulated response incorporating input and parameter uncertainty
  Y_sim <- exp(beta0_sim + beta1_sim * log(X1_sim) + beta2_sim * log(X2_sim))   
  
  # Add stochastic noise to reflect target uncertainty
  Y_simulated[, i] <- Y_sim * rlnorm(n, meanlog = 0, sdlog = 0.1)
}

# Compute mean and confidence intervals
data$Y_mean_sim <- apply(Y_simulated, 1, mean)
data$Y_lower <- apply(Y_simulated, 1, quantile, probs = 0.025)
data$Y_upper <- apply(Y_simulated, 1, quantile, probs = 0.975)

# Compute R² for stochastic ensemble mean model
R2_stochastic <- cor(log(data$Y_mean_sim), log(data$Y_true))^2

### Step 4: Visualization ###

# Plot 1: Deterministic Model
plot1 <- ggplot(data, aes(x = log(Y_true), y = log(Y_pred))) +
  geom_point(color = "blue", alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", color = "blue", se = FALSE, size = 1.2) +
  labs(
    title = "Deterministic Model",
    x = "Log(True Y)", y = "Log(Predicted Y)",
    subtitle = sprintf("R² Deterministic: %.3f", R2_deterministic)
  ) +
  theme_minimal() 

# Plot 2: Stochastic Ensemble Model
plot2 <- ggplot(data, aes(x = log(Y_true), y = log(Y_mean_sim))) +
  geom_point(color = "red", alpha = 0.5, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE, size = 1.2) +
  geom_ribbon(aes(ymin = log(Y_lower), ymax = log(Y_upper)), fill = "gray", alpha = 0.2) +
  labs(
    title = "Stochastic Ensemble Model",
    x = "Log(True Y)", y = "Log(Ensemble Mean Y)",
    subtitle = sprintf("R² Stochastic: %.3f", R2_stochastic)
  ) +
  theme_minimal() 

# Plot 3: Comparison of Deterministic vs. Stochastic Predictions
plot3 <- ggplot(data, aes(x = log(Y_true))) +
  geom_point(aes(y = log(Y_pred)), color = "blue", alpha = 0.5, size = 2) +  # Deterministic Predictions
  geom_point(aes(y = log(Y_mean_sim)), color = "red", alpha = 0.5, size = 2) +  # Stochastic Mean
  geom_smooth(aes(y = log(Y_pred)), method = "lm", color = "blue", se = FALSE, size = 1.2) +  # Det. Best Fit Line
  geom_smooth(aes(y = log(Y_mean_sim)), method = "lm", color = "red", se = FALSE, size = 1.2) +  # Ensemble Mean Line
  labs(
    title = "Comparison of Deterministic vs. Stochastic Predictions",
    x = "Log(True Y)", y = "Log(Predicted Y)",
    subtitle = sprintf("R² Deterministic: %.3f | R² Stochastic: %.3f", R2_deterministic, R2_stochastic)
  ) +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        plot.subtitle = element_text(size = 12, face = "italic"))

# Display all three plots
print(plot1)
print(plot2)
print(plot3)


# Display all the realizations
# Number of stochastic realizations
num_realizations <- 100

# Generate stochastic ensemble data
set.seed(123)
ensemble_data <- data.frame()

for (i in 1:num_realizations) {
  # Simulating Y values with stochasticity
  Y_sim <- exp(rnorm(nrow(data), mean = log(data$Y_mean_sim), sd = 0.3))
  
  # Store each realization with an identifier
  ensemble_data <- rbind(ensemble_data, data.frame(
    Y_true = data$Y_true,  # Keeping the same true Y values
    Y_sim = Y_sim,         # Simulated Y for this realization
    realization = i        # Realization identifier
  ))
}

print(plot2)