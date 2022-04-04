// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_GL_CELL_H
#define Q_HYODA_GL_CELL_H

#include <QtWidgets>
#include <QHyodaGL.h>
#include <QVector>
#include <QVector3D> 

class QHyodaGLCell:public QHyodaGL{
public:
  QHyodaGLCell(QWidget*);
  ~QHyodaGLCell(){qDebug()<<"~QHyodaGLCell";}
public:
  void clear();
  void add_node(const QVector3D&, const QColor);
  void draw() const;
public:
  void set_hexahedron_nodes(QVector<QVector3D>&);
  void set_triangle_nodes(QVector<QVector3D>&);
  void set_quads_nodes(QVector<QVector3D> &);
public:
  quint64 cell_nb_nodes;
  QVector<QVector3D> nodes;
  QVector<QVector4D> colors;
private:
  const GLubyte stdIdx[8] = {0,1,2,3,4,5,6,7};
  const GLubyte hexIdx[24] = {0,3,2,1,2,3,7,6,0,4,7,3,1,2,6,5,4,5,6,7,0,1,5,4};

};

#endif // Q_HYODA_GL_CELL_H
