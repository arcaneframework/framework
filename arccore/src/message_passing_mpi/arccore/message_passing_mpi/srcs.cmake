
set(SOURCES
  MessagePassingMpiGlobal.h
  MessagePassingMpiGlobal.cc
  MessagePassingMpiEnum.cc
  MpiAdapter.cc
  MpiDatatype.h
  MpiDatatype.cc
  MpiControlDispatcher.cc
  MpiRequestList.cc
  MpiSerializeDispatcher.cc
  MpiTypeDispatcher.cc
  MpiMessagePassingMng.cc
  MpiMessagePassingMng.h
  MpiContigMachineShMemWinBaseInternal.cc
  MpiContigMachineShMemWinBaseInternalCreator.cc
  MpiMachineShMemWinBaseInternal.cc
  MpiMultiMachineShMemWinBaseInternal.cc

  StandaloneMpiMessagePassingMng.cc
  StandaloneMpiMessagePassingMng.h

  internal/IMpiProfiling.h
  internal/MessagePassingMpiEnum.h
  internal/MpiAdapter.h
  internal/MpiControlDispatcher.h
  internal/MpiLock.h
  internal/MpiContigMachineShMemWinBaseInternal.h
  internal/MpiContigMachineShMemWinBaseInternalCreator.h
  internal/MpiMachineShMemWinBaseInternal.h
  internal/MpiMultiMachineShMemWinBaseInternal.h
  internal/MpiRequest.h
  internal/MpiRequestList.h
  internal/MpiSerializeDispatcher.h
  internal/MpiTypeDispatcher.h
  internal/MpiTypeDispatcherImpl.h
  internal/NoMpiProfiling.h
)
