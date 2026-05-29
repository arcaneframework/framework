set(ARCANE_SOURCES
  ArcaneThread.h
  ArcaneThreadMisc.h

  GlibThreadImplementation.cc
  GlibThreadMng.h

  IAsyncQueue.h
  AsyncQueue.cc

  ISharedMemoryMessageQueue.h
  SharedMemoryParallelSuperMng.cc
  SharedMemoryParallelSuperMng.h
  SharedMemoryParallelDispatch.cc
  SharedMemoryParallelDispatch.h
  SharedMemoryMessageQueue.cc
  SharedMemoryMessageQueue.h
  SharedMemoryParallelMng.cc
  SharedMemoryParallelMng.h

  internal/SharedMemoryThreadMng.h
  internal/SharedMemoryContigMachineShMemWinBaseInternal.cc
  internal/SharedMemoryContigMachineShMemWinBaseInternal.h
  internal/SharedMemoryMachineShMemWinBaseInternal.cc
  internal/SharedMemoryMachineShMemWinBaseInternal.h
  internal/SharedMemoryMachineShMemWinBaseInternalCreator.cc
  internal/SharedMemoryMachineShMemWinBaseInternalCreator.h

  # TODO: the following files are kept for compatibility with the existing system. They must be removed
  IThreadMessageQueue.h
  ThreadMessageQueue2.h
  ThreadParallelMng.h
  ThreadParallelDispatch.h
)
