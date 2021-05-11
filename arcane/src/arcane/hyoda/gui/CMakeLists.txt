arcane_find_package(Qt5 5.7.0 QUIET COMPONENTS Core OpenGL Network Widgets)

message(STATUS "HYODA: Qt5Core_DIR is ${Qt5Core_DIR}")

if (TARGET Qt5::Core AND TARGET Qt5::OpenGL AND TARGET Qt5::Network AND TARGET Qt5::Widgets)

  message(STATUS "HYODA: all requisites are found")

  set(HYODA_UI hyoda.ui hyodaCell.ui hyodaJob.ui hyodaMatrix.ui hyodaMesh.ui hyodaPapi.ui)
  QT5_WRAP_UI(HYODA_UI_H ${HYODA_UI})
   
  QT5_ADD_RESOURCES(HYODA_QRC hyoda.qrc)
  set(HYODA_CC main.cc QHyoda.cc
    QHyodaGL.cc QHyodaGLCell.cc QHyodaIceT.cc
    QHyodaMachine.cc QHyodaGdb.cc QHyodaJob.cc 
    QHyodaTool.cc QHyodaToolCell.cc QHyodaToolMesh.cc
	  QHyodaSsh.cc QHyodaTcp.cc QHyodaToolMatrix.cc
    QHyodaX11.cc QHyodaX11Xterm.cc
    QHyodaX11XtermLog.cc QHyodaPapi.cc)

  QT5_WRAP_CPP(HYODA_MOC
    QHyoda.h
    QHyodaGL.h
    QHyodaIceT.h
    QHyodaMachine.h
    QHyodaGdb.h
    QHyodaJob.h
    QHyodaTool.h
    QHyodaToolCell.h
    QHyodaToolMesh.h
    QHyodaSsh.h
    QHyodaTcp.h
    QHyodaToolMatrix.h
    QHyodaX11.h
    QHyodaX11Xterm.h
    QHyodaX11XtermLog.h
    QHyodaPapi.h
    )

  add_executable(hyoda ${HYODA_QRC} ${HYODA_MOC} ${HYODA_UI_H} ${HYODA_CC})
  set_target_properties(hyoda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  target_link_libraries(hyoda Qt5::Widgets Qt5::Core Qt5::OpenGL Qt5::Network GL)
  target_include_directories(hyoda PRIVATE ${ARCANE_SRC_PATH}/arcane/hyoda/gui ${CMAKE_CURRENT_BINARY_DIR})
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hyoda DESTINATION bin)
endif()
