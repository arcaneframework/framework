find_package(Glib REQUIRED)

set(SOURCES
  ArccoreGlobal.h
  ArccoreGlobal.cc
  APReal.h
  ArrayIterator.h
  ArrayRange.h
  ArrayView.h
  ArrayView.cc
  Array2View.h
  Array3View.h
  Array4View.h
  ArgumentException.cc
  ArgumentException.h
  CheckedPointer.h
  CStringUtils.h
  CStringUtils.cc
  BaseTypes.h
  BasicTranscoder.h
  BasicTranscoder.cc
  BaseTypes.h
  CoreArray.h
  Exception.h
  Exception.cc
  FatalErrorException.cc
  FatalErrorException.h
  Iterator.h
  IndexOutOfRangeException.cc
  IndexOutOfRangeException.h
  IStackTraceService.h
  NotSupportedException.h
  NotSupportedException.cc
  NotImplementedException.h
  NotImplementedException.cc
  PlatformUtils.h
  PlatformUtils.cc
  ReferenceCounter.h
  Span.h
  Span2.h
  StackTrace.h
  StackTrace.cc
  String.h
  String.cc
  StringBuilder.h
  StringBuilder.cc
  StringImpl.h
  StringImpl.cc
  TraceInfo.h
  TimeoutException.h
  TimeoutException.cc

  IFunctor.h
  Functor.h
  Functor.cc
)

arccore_add_component_library(base
  FILES ${SOURCES}
)

target_link_libraries(arccore_base PUBLIC $<BUILD_INTERFACE:arccore_build_compile_flags>)

target_link_libraries(arccore_base PRIVATE arcconpkg_Glib)

target_compile_features(arccore_base PUBLIC cxx_std_17)

# Nécessaire pour pouvoir inclure 'arccore_config.h'
target_include_directories(arccore_base PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
