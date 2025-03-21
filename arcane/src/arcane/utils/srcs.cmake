set(ARCANE_SOURCES
  ApplicationInfo.cc
  ApplicationInfo.h
  ArcaneCxx20.h
  ArcaneGlobal.cc
  ArcaneGlobal.h
  ArithmeticException.cc
  ArithmeticException.h
  Array.cc
  Array.h
  ArrayShape.cc
  ArrayShape.h
  ArraySimdPadder.h
  Atomic.cc
  Atomic.h
  AMRCallBackMng.cc
  AMRCallBackMng.h
  BasicDataType.h
  BadAlignmentException.cc
  BadAlignmentException.h
  BadCastException.cc
  BadCastException.h
  BFloat16.h
  Collection.cc
  Collection.h
  CommandLineArguments.h
  CommandLineArguments.cc
  Convert.cc
  Convert.h
  ConcurrencyUtils.cc
  ConcurrencyUtils.h
  CStringUtils.cc
  CStringUtils.h
  DependencyInjection.cc
  DualUniqueArray.h
  Enumerator.cc
  Enumerator.h
  ExternalRef.h
  Exception.h
  Exception.cc
  ExtentsV.h
  Event.cc
  Event.h
  FixedArray.h
  Float16.h
  Float128.h
  FloatingPointExceptionSentry.cc
  FloatingPointExceptionSentry.h
  FileContent.cc
  FileContent.h
  ForLoopTraceInfo.cc
  ForLoopTraceInfo.h
  ForLoopRanges.h
  GenericRegisterer.h
  GenericRegisterer.cc
  GoBackwardException.cc
  GoBackwardException.h
  HashSuite.h
  IDataCompressor.h
  Int128.h
  IOException.cc
  IOException.h
  IMemoryRessourceMng.h
  JSONPropertyReader.h
  JSONReader.cc
  JSONReader.h
  JSONWriter.cc
  JSONWriter.h
  LinearOffsetMap.h
  LinearOffsetMap.cc
  HashAlgorithm.cc
  HashTable.cc
  HashTable.h
  HPReal.cc
  HPReal.h
  InvalidArgumentException.cc
  InvalidArgumentException.h
  IMessagePassingProfilingService.h
  ISO88591Transcoder.cc
  ISO88591Transcoder.h
  MDSpan.h
  MemoryAllocator.h
  MemoryPool.cc
  MemoryView.h
  MemoryView.cc
  Misc.cc
  MD5HashAlgorithm.cc
  MD5HashAlgorithm.h
  MDDim.h
  Math.cc
  Math.h
  MemoryAccessInfo.cc
  MemoryAccessInfo.h
  MemoryBuffer.cc
  MemoryInfo.cc
  MemoryInfo.h
  MemoryRessource.h
  MemoryResourceMng.cc
  MemoryUtils.h
  MemoryUtils.cc
  Numeric.cc
  Numeric.h
  NumericTypes.h
  NumArray.h
  NumArrayContainer.h
  NumArrayUtils.h
  NumericTraits.h
  NumMatrix.h
  NumVector.h
  Observable.cc
  Observer.cc
  Observable.h
  Observer.h
  OStringStream.cc
  OStringStream.h
  ParallelFatalErrorException.cc
  ParallelFatalErrorException.h
  ParallelLoopOptions.h
  ParallelLoopOptions.cc
  ParameterCaseOption.h
  ParameterCaseOption.cc
  PerfCounterMng.cc
  PerfCounterMng.h
  PlatformUtils.cc
  PlatformUtils.h
  Process.cc
  Process.h
  Profiling.h
  Profiling.cc
  Property.cc
  Property.h
  PropertyDeclarations.h
  Ptr.cc
  Ptr.h
  Ref.h
  Real2.cc
  Real2.h
  Real2x2.cc
  Real2x2.h
  Real3.cc
  Real3.h
  Real3x3.cc
  Real3x3.h
  SignalException.cc
  SignalException.h
  Simd.cc
  Simd.h
  SmallArray.cc
  SmallArray.h
  TestLogger.h
  TestLogger.cc
  TraceAccessor2.h
  TraceAccessor2.cc
  TraceMng.cc
  UserDataList.cc
  UserDataList.h
  ValueChecker.cc
  ValueChecker.h
  Vector2.h
  Vector3.h
  VersionInfo.cc
  VersionInfo.h
  ApplicationInfo.h
  ArcaneGlobal.h
  ArrayBoundsIndex.h
  ArrayExtents.h
  ArrayExtentsValue.h
  ArrayBounds.h
  ArrayIndex.h
  MDIndex.h
  ArrayLayout.h
  LoopRanges.h
  ArithmeticException.h
  Array.h
  Atomic.h
  AMRCallBackMng.h
  BadAlignmentException.h
  BadCastException.h
  Collection.h
  Convert.h
  CStringUtils.h
  Enumerator.h
  FloatingPointExceptionSentry.h
  FileContent.h
  GoBackwardException.h
  IOException.h
  JSONReader.h
  JSONWriter.h
  HashTable.h
  HPReal.h
  InvalidArgumentException.h
  ISO88591Transcoder.h
  MD5HashAlgorithm.h
  Math.h
  MemoryAccessInfo.h
  MemoryInfo.h
  Numeric.h
  OStringStream.h
  ParallelFatalErrorException.h
  PerfCounterMng.h
  PlatformUtils.h
  Process.h
  Ptr.h
  Real2.h
  Real2x2.h
  Real3.h
  Real3x3.h
  SignalException.h
  Simd.h
  UserDataList.h
  ValueChecker.h
  VersionInfo.h
  ArcanePrecomp.h
  ArgumentException.h
  Array2.h
  ArrayUtils.h
  ArrayView.h
  Array2View.h
  Array3View.h
  Array4View.h
  ArrayIterator.h
  ArrayRange.h
  AutoDestroyUserData.h
  DataTypeContainer.h
  APReal.h
  BasicTranscoder.h
  CoreArray.h
  MultiArray2.h
  MultiArray2View.h
  ArrayImpl.h
  ArrayConverter.h
  AutoRef.h
  BuiltInProxy.h
  CheckedConvert.h
  UtilsTypes.h
  CollectionImpl.h
  CriticalSection.h
  Deleter.h
  EventHandler.h
  EventHandlerList.h
  FatalErrorException.h
  Functor.h
  FunctorUtils.h
  FunctorWithAddress.h
  FunctorWithArgument.h
  HashFunction.h
  HashFunction.cc
  IndexOutOfRangeException.h
  ItemGroupObserver.h
  IDynamicLibraryLoader.h
  IFunctor.h
  IFunctorWithAddress.h
  IMathFunctor.h
  IFunctorWithArgument.h
  IHashAlgorithm.h
  IObserver.h
  IObservable.h
  Iterator.h
  IMemoryAllocator.h
  IRangeFunctor.h
  ITraceMng.h
  ITraceMngPolicy.h
  Iostream.h
  IOnlineDebuggerService.h
  IPerformanceCounterService.h
  IProfilingService.h
  IProcessorAffinityService.h
  IStackTraceService.h
  ISymbolizerService.h
  IUserData.h
  IUserDataList.h
  IThreadBarrier.h
  IThreadImplementation.h
  IThreadImplementationService.h
  IThreadMng.h
  ITranscoder.h
  Limits.h
  List.h
  ListImpl.h
  MathApfloat.h
  MultiBuffer.h
  Mutex.h
  NotImplementedException.h
  NotSupportedException.h
  NullThreadMng.h
  HashTableMap.h
  HashTableMap2.h
  HashTableMap2.cc
  ObjectImpl.h
  ParameterList.h
  ParameterList.cc
  RangeFunctor.h
  Real2Proxy.h
  Real2x2Proxy.h
  Real3Proxy.h
  Real3x3Proxy.h
  StdHeader.h
  SimdCommon.h
  SimdEMUL.h
  SimdEMULGenerated.h
  SimdAVX.h
  SimdAVXGenerated.h
  SimdAVX512.h
  SimdAVX512Generated.h
  SimdSSE.h
  SimdSSEGenerated.h
  SimdOperation.h
  SpinLock.h
  SharedArray.h
  StackTrace.h
  StringImpl.h
  String.h
  StringBuilder.h
  StringDictionary.h
  StringDictionary.cc
  StringList.h
  SHA1HashAlgorithm.h
  SHA1HashAlgorithm.cc
  SHA3HashAlgorithm.h
  SHA3HashAlgorithm.cc
  ValueConvert.h
  ValueConvert.cc
  ScopedPtr.h
  SharedPtr.h
  EventHandlerListImpl.h
  HashTableSet.h
  NameComparer.h
  TimeoutException.h
  TraceInfo.h
  Trace.h
  TraceAccessor.h
  TraceMessage.h
  TraceClassConfig.h
  UniqueArray.h
  IMemoryInfo.h
  AMRComputeFunction.h
  IAMRTransportFunctor.h
  AMRTransportFunctor.h
  GraphBaseT.h
  DirectedGraphT.h
  DirectedAcyclicGraphT.h
  internal/DependencyInjection.h
  internal/ApplicationInfoProperties.h
  internal/MemoryResourceMng.h
  internal/MemoryUtilsInternal.h
  internal/IMemoryRessourceMngInternal.h
  internal/IMemoryCopier.h
  internal/ParameterOption.h
  internal/ParameterOption.cc
  internal/ProfilingInternal.h
  internal/ValueConvertInternal.h
  internal/SpecificMemoryCopyList.h
  internal/MemoryBuffer.h
  internal/MemoryPool.h
  internal/ParallelLoopOptionsProperties.h
  internal/TaskFactoryInternal.h
  )

if (ARCANE_HAS_CXX20)
  list(APPEND ARCANE_SOURCES
    ArcaneCxx20.cc
    )
endif()

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    MDSpan.cc
    NumArray.cc
    DualUniqueArray.cc
    NumArrayUtils.cc
  )
endif()
