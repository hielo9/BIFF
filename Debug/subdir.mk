################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../particleSystem.cpp \
../particles.cpp \
../render_particles.cpp \
../shaders.cpp 

CU_SRCS += \
../particleSystem_cuda.cu 

CU_DEPS += \
./particleSystem_cuda.d 

OBJS += \
./particleSystem.o \
./particleSystem_cuda.o \
./particles.o \
./render_particles.o \
./shaders.o 

CPP_DEPS += \
./particleSystem.d \
./particles.d \
./render_particles.d \
./shaders.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/workspace/BIFF" -O3 -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/workspace/BIFF" -O3 -m64 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/workspace/BIFF" -O3 -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/workspace/BIFF" -O3 -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


