#ifndef Q_HYODA_JOB_TOOL_MESH_H
#define Q_HYODA_JOB_TOOL_MESH_H

#include <QtWidgets>
#include "QHyodaIceT.h"
#include "ui_hyodaMesh.h"

class QHyodaToolMesh: public QWidget, public  Ui::toolMeshWidget{
  Q_OBJECT
public:
  QHyodaToolMesh(QTabWidget*);
  ~QHyodaToolMesh();
};
#endif //  Q_HYODA_JOB_TOOL_MESH_H
