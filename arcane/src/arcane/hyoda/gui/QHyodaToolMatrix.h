#ifndef Q_HYODA_JOB_TOOL_MATRIX_H
#define Q_HYODA_JOB_TOOL_MATRIX_H

#include <QtWidgets>
#include "QHyodaIceT.h"
#include "ui_hyodaMatrix.h"

class QHyodaToolMatrix: public QWidget, public  Ui::toolMatrixWidget{
  Q_OBJECT
public:
  QHyodaToolMatrix(QTabWidget*);
  ~QHyodaToolMatrix();
};
#endif //  Q_HYODA_JOB_TOOL_MATRIX_H
