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
  internal/SharedMemoryContigMachineShMemWinBaseInternalCreator.cc
  internal/SharedMemoryContigMachineShMemWinBaseInternalCreator.h

  # TODO: les fichiers suivants sont gardés pour des raisons
  # de compatibilité avec l'existant. Il faudra les supprimer
  IThreadMessageQueue.h
  ThreadMessageQueue2.h
  ThreadParallelMng.h
  ThreadParallelDispatch.h
)
