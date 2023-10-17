#!/bin/bash

echo "Gheorghica Radu-Iulian, ChatGPT, 2023"

# Display a menu
echo "Select a script to run:"
echo "1. Run QCC_colored_points.py"
echo "2. Run script2.py"
echo "3. Run script3.py"
echo "4. Exit"

# Get user input
read -p "Enter your choice: " choice

case $choice in
  1)
    echo "Running QCC_colored_points.py"
    python /app/Colored_points_hybrid_classifier/QCC_colored_points.py
    ;;
  2)
    echo "Running script2.py"
    python script2.py
    ;;
  3)
    echo "Running script3.py"
    python script3.py
    ;;
  4)
    echo "Exiting"
    ;;
  *)
    echo "Invalid choice"
    ;;
esac





