include(srcs.cmake)

arccore_add_component_library(accelerator
  FILES ${SOURCES}
)

arccore_accelerator_add_source_files(
  MemoryCopier.cc
  MemoryCopierTpl1.cc
  MemoryCopierTpl2.cc
  MemoryCopierTpl3.cc
  MemoryCopierTpl4.cc
  CommonUtils.cc
)

target_link_libraries(arccore_accelerator PUBLIC
  arccore_common
  arccore_concurrency
  "$<TARGET_NAME_IF_EXISTS:Arccore::arccore_accelerator_cuda>"
  "$<TARGET_NAME_IF_EXISTS:Arccore::arccore_accelerator_hip>"
  "$<TARGET_NAME_IF_EXISTS:Arccore::arccore_accelerator_sycl>"
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
