# Load necessary library
library(ggplot2)

# Seed for reproducibility
set.seed(123)

# Generate continuous data (normal distribution with mean = 50, sd = 10)
continuous_data <- rnorm(1000, mean = 50, sd = 10)

# Adjusting parameters to approximate the normal distribution shape with a Poisson distribution
# Using the mean of the continuous data as lambda for the Poisson distribution
lambda <- mean(continuous_data)
discrete_data <- rpois(1000, lambda = lambda)

# Calculate quantiles for both datasets
quantiles_continuous <- quantile(continuous_data, probs = c(0.25, 0.5, 0.75))
quantiles_discrete <- quantile(discrete_data, probs = c(0.25, 0.5, 0.75))

# Create a combined data frame for plotting
data_continuous <- data.frame(Value = continuous_data, Type = "Continuous")
data_discrete <- data.frame(Value = discrete_data, Type = "Discrete")
combined_data <- rbind(data_continuous, data_discrete)

# Plotting
ggplot() +
  geom_density(data = data_continuous, aes(x = Value, fill = "blue", y = ..scaled..),
               alpha = 0.5, adjust = 1.5, color = "black") +
  geom_histogram(data = data_discrete, aes(x = Value, fill = "red", y = ..density..),
                 bins = 30, alpha = 0.5, position = "identity", color = "black") +
  geom_vline(data = data.frame(Type = "Continuous", xintercept = quantiles_continuous),
             aes(xintercept = xintercept), col = "blue", linetype = "dashed") +
  geom_vline(data = data.frame(Type = "Discrete", xintercept = quantiles_discrete),
             aes(xintercept = xintercept), col = "red", linetype = "dashed") +
  labs(title = "Density and Histogram for Continuous and Discrete Data",
       x = "Value", y = "Density") +
  facet_wrap(~Type, scales = "free_y") +
  theme_minimal()

# Print quantiles and histograms to console
cat("Quantiles for Continuous Data:\n")
print(quantiles_continuous)
cat("Quantiles for Discrete Data:\n")
print(quantiles_discrete)