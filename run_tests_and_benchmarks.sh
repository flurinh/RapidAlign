#!/bin/bash

# Run Tests and Benchmarks for Batch Point Cloud Alignment

# Exit on error
set -e

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set up work directory
WORK_DIR=$(pwd)
BUILD_DIR="$WORK_DIR/build"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}   Running Tests and Benchmarks for Point Cloud Alignment   ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Make sure we have a build directory
echo -e "${YELLOW}Checking build directory...${NC}"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${GREEN}Creating build directory...${NC}"
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

# Run cmake if not already configured
if [ ! -f "$BUILD_DIR/Makefile" ]; then
    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cmake ..
else
    echo -e "${GREEN}Build already configured.${NC}"
fi

# Build the test programs
echo -e "${YELLOW}Building test programs...${NC}"
make -j4

# Define test programs
TEST_PROGRAMS=("TestBatchAlign" "CPUTest" "GraphAlignTest")

# Run each test program
for prog in "${TEST_PROGRAMS[@]}"; do
    if [ -f "$BUILD_DIR/$prog" ]; then
        echo -e "${YELLOW}Running $prog...${NC}"
        ./$prog
        echo -e "${GREEN}$prog completed successfully!${NC}"
    else
        echo -e "${RED}Error: $prog not found!${NC}"
    fi
done

# Run visualization script if Python is available
echo -e "${YELLOW}Checking for Python visualization script...${NC}"
if [ -f "$WORK_DIR/visualize_graphs.py" ]; then
    echo -e "${YELLOW}Running visualization script...${NC}"
    python3 "$WORK_DIR/visualize_graphs.py"
    echo -e "${GREEN}Visualization script completed!${NC}"
else
    echo -e "${RED}Warning: visualization_graphs.py not found. Skipping visualization.${NC}"
fi

# Run benchmark visualization script if available
echo -e "${YELLOW}Checking for benchmark visualization scripts...${NC}"
if [ -f "$WORK_DIR/plot_optimization_benchmarks.py" ]; then
    echo -e "${YELLOW}Running benchmark visualization...${NC}"
    python3 "$WORK_DIR/plot_optimization_benchmarks.py"
    echo -e "${GREEN}Benchmark visualization completed!${NC}"
else
    echo -e "${RED}Warning: plot_optimization_benchmarks.py not found. Skipping benchmark visualization.${NC}"
fi

# Run traditional plot script if available
if [ -f "$BUILD_DIR/plot_benchmarks.py" ]; then
    echo -e "${YELLOW}Running traditional benchmark visualization...${NC}"
    python3 "$BUILD_DIR/plot_benchmarks.py"
    echo -e "${GREEN}Traditional benchmark visualization completed!${NC}"
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}All tests and benchmarks completed successfully!${NC}"
echo -e "${BLUE}=========================================================${NC}"

# List generated files
echo -e "${YELLOW}Generated files:${NC}"
find "$BUILD_DIR" -name "*.png" -o -name "*.csv" -o -name "*.ply" | sort