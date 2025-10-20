if(ARCCORE_ENABLE_GLIB)
  find_package(Glib 2.32 REQUIRED)
endif()

set(SOURCES
  ArccoreGlobal.h
  ArccoreGlobal.cc
  APReal.h
  ArrayBounds.h
  ArrayIterator.h
  ArrayRange.h
  ArrayViewCommon.h
  ArrayView.h
  ArrayView.cc
  Array2View.h
  Array3View.h
  Array4View.h
  ArrayExtents.h
  ArrayExtentsValue.h
  ArrayLayout.h
  ArgumentException.cc
  ArgumentException.h
  BuiltInDataTypeContainer.h
  BFloat16.h
  CheckedPointer.h
  CStringUtils.h
  CStringUtils.cc
  BaseTypes.h
  BasicDataType.h
  BasicDataType.cc
  BasicTranscoder.h
  BasicTranscoder.cc
  BaseTypes.h
  CoreArray.h
  Exception.h
  Exception.cc
  ExternalRef.h
  ExternalRef.cc
  ExtentsV.h
  FatalErrorException.cc
  FatalErrorException.h
  Float16.h
  Float128.h
  FloatConversion.h
  FixedArray.h
  Iterator.h
  IndexOutOfRangeException.cc
  IndexOutOfRangeException.h
  Int128.h
  IStackTraceService.h
  MDDim.h
  MDIndex.h
  MDSpan.h
  MDSpan.cc
  MemoryView.h
  MemoryView.cc
  NotSupportedException.h
  NotSupportedException.cc
  NotImplementedException.h
  NotImplementedException.cc
  NumericTraits.h
  PlatformUtils.h
  PlatformUtils.cc
  ReferenceCounter.h
  ReferenceCounterImpl.h
  Ref.h
  RefBase.h
  RefDeclarations.h
  Ref.cc
  Span.h
  Span2.h
  StackTrace.h
  StackTrace.cc
  String.h
  String.cc
  StringUtils.h
  StringBuilder.h
  StringBuilder.cc
  StringImpl.h
  StringImpl.cc
  StringView.h
  StringView.cc
  TraceInfo.h
  TimeoutException.h
  TimeoutException.cc
  ValueFiller.h

  IFunctor.h
  Functor.h
  Functor.cc

  internal/StringImpl.h
)

arccore_add_component_library(base
  FILES ${SOURCES}
)

target_link_libraries(arccore_base PUBLIC $<BUILD_INTERFACE:arccore_build_compile_flags>)

if(ARCCORE_ENABLE_GLIB)
  target_link_libraries(arccore_base PRIVATE arcconpkg_Glib)
  target_compile_definitions(arccore_base PRIVATE ARCCORE_HAS_GLIB)
endif()

target_compile_features(arccore_base PUBLIC cxx_std_${ARCCORE_CXX_STANDARD})

# Nécessaire pour pouvoir inclure 'arccore_config.h'
target_include_directories(arccore_base PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
