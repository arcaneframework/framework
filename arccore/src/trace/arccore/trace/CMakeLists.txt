set(SOURCES
  TraceGlobal.h
  TraceGlobal.cc
  Trace.h
  TraceAccessor.h
  TraceAccessor.cc
  TraceClassConfig.h
  TraceMessage.h
  TraceMessage.cc
  ITraceMng.h
  TraceMng.cc
  ITimeMetricCollector.h
  TimeMetric.h
  TimeMetric.cc
  StandaloneTraceMessage.h
)

arccore_add_component_library(trace
  FILES ${SOURCES}
)

target_link_libraries(arccore_trace PUBLIC arccore_base)
target_link_libraries(arccore_trace PRIVATE arccore_concurrency)
