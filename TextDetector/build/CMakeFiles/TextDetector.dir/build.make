# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/minhnd/FTI.Projects/fti_id/TextDetector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/minhnd/FTI.Projects/fti_id/TextDetector/build

# Include any dependencies generated for this target.
include CMakeFiles/TextDetector.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TextDetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TextDetector.dir/flags.make

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o: ../ImageBinarization.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/ImageBinarization.cpp

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/ImageBinarization.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/ImageBinarization.cpp > CMakeFiles/TextDetector.dir/ImageBinarization.cpp.i

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/ImageBinarization.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/ImageBinarization.cpp -o CMakeFiles/TextDetector.dir/ImageBinarization.cpp.s

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.requires

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.provides: CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.provides

CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.provides.build: CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o


CMakeFiles/TextDetector.dir/TextDetector.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/TextDetector.cpp.o: ../TextDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/TextDetector.dir/TextDetector.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/TextDetector.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/TextDetector.cpp

CMakeFiles/TextDetector.dir/TextDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/TextDetector.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/TextDetector.cpp > CMakeFiles/TextDetector.dir/TextDetector.cpp.i

CMakeFiles/TextDetector.dir/TextDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/TextDetector.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/TextDetector.cpp -o CMakeFiles/TextDetector.dir/TextDetector.cpp.s

CMakeFiles/TextDetector.dir/TextDetector.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/TextDetector.cpp.o.requires

CMakeFiles/TextDetector.dir/TextDetector.cpp.o.provides: CMakeFiles/TextDetector.dir/TextDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/TextDetector.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/TextDetector.cpp.o.provides

CMakeFiles/TextDetector.dir/TextDetector.cpp.o.provides.build: CMakeFiles/TextDetector.dir/TextDetector.cpp.o


CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o: ../BlobsLib/blob.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/blob.cpp

CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/blob.cpp > CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.i

CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/blob.cpp -o CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.s

CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.requires

CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.provides: CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.provides

CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.provides.build: CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o


CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o: ../BlobsLib/BlobContour.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobContour.cpp

CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobContour.cpp > CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.i

CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobContour.cpp -o CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.s

CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.requires

CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.provides: CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.provides

CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.provides.build: CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o


CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o: ../BlobsLib/BlobOperators.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobOperators.cpp

CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobOperators.cpp > CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.i

CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobOperators.cpp -o CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.s

CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.requires

CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.provides: CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.provides

CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.provides.build: CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o


CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o: ../BlobsLib/BlobResult.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobResult.cpp

CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobResult.cpp > CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.i

CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/BlobResult.cpp -o CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.s

CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.requires

CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.provides: CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.provides

CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.provides.build: CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o


CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o: CMakeFiles/TextDetector.dir/flags.make
CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o: ../BlobsLib/ComponentLabeling.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o -c /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/ComponentLabeling.cpp

CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/ComponentLabeling.cpp > CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.i

CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/minhnd/FTI.Projects/fti_id/TextDetector/BlobsLib/ComponentLabeling.cpp -o CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.s

CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.requires:

.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.requires

CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.provides: CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.requires
	$(MAKE) -f CMakeFiles/TextDetector.dir/build.make CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.provides.build
.PHONY : CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.provides

CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.provides.build: CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o


# Object files for target TextDetector
TextDetector_OBJECTS = \
"CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o" \
"CMakeFiles/TextDetector.dir/TextDetector.cpp.o" \
"CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o" \
"CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o" \
"CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o" \
"CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o" \
"CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o"

# External object files for target TextDetector
TextDetector_EXTERNAL_OBJECTS =

libTextDetector.so: CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/TextDetector.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o
libTextDetector.so: CMakeFiles/TextDetector.dir/build.make
libTextDetector.so: /usr/local/lib/libopencv_cudastereo.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_superres.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_videostab.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_stitching.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_face.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_saliency.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_stereo.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_freetype.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_bgsegm.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_tracking.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_optflow.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_dpm.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_surface_matching.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_ccalib.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_img_hash.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_ximgproc.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_bioinspired.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_structured_light.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_rgbd.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_reg.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_plot.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_aruco.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_hfs.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_xphoto.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_fuzzy.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudacodec.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_shape.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudawarping.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_photo.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudafilters.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_datasets.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_text.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_ml.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_dnn.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_video.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_objdetect.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_calib3d.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_features2d.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_flann.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_highgui.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_videoio.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_imgproc.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_core.so.3.4.1
libTextDetector.so: /usr/local/lib/libopencv_cudev.so.3.4.1
libTextDetector.so: CMakeFiles/TextDetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library libTextDetector.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TextDetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TextDetector.dir/build: libTextDetector.so

.PHONY : CMakeFiles/TextDetector.dir/build

CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/ImageBinarization.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/TextDetector.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/BlobsLib/blob.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/BlobsLib/BlobContour.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/BlobsLib/BlobOperators.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/BlobsLib/BlobResult.cpp.o.requires
CMakeFiles/TextDetector.dir/requires: CMakeFiles/TextDetector.dir/BlobsLib/ComponentLabeling.cpp.o.requires

.PHONY : CMakeFiles/TextDetector.dir/requires

CMakeFiles/TextDetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TextDetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TextDetector.dir/clean

CMakeFiles/TextDetector.dir/depend:
	cd /home/minhnd/FTI.Projects/fti_id/TextDetector/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/minhnd/FTI.Projects/fti_id/TextDetector /home/minhnd/FTI.Projects/fti_id/TextDetector /home/minhnd/FTI.Projects/fti_id/TextDetector/build /home/minhnd/FTI.Projects/fti_id/TextDetector/build /home/minhnd/FTI.Projects/fti_id/TextDetector/build/CMakeFiles/TextDetector.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TextDetector.dir/depend

