// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_GL_H
#define Q_HYODA_GL_H

#include <QVector4D> 
#include <QVector2D>
#include <QOpenGLWidget>
#include <QWheelEvent>

class QHyodaGL: public QOpenGLWidget{
  Q_OBJECT
public:
  explicit QHyodaGL(QWidget* = 0);
  ~QHyodaGL(){}
  virtual void draw() const =0;
protected:
  void initializeGL() Q_DECL_OVERRIDE;
  void resizeGL(int,int) Q_DECL_OVERRIDE;
  void paintGL() Q_DECL_OVERRIDE;
protected:
  void mousePressEvent(QMouseEvent*) Q_DECL_OVERRIDE;
  void mouseMoveEvent(QMouseEvent*) Q_DECL_OVERRIDE;
  void wheelEvent(QWheelEvent*) Q_DECL_OVERRIDE;
public:
  QVector2D mouse;
  QVector4D pov;
};

#endif // Q_HYODA_GL_H
