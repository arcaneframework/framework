// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef Q_HYODA_ICE_T_H
#define Q_HYODA_ICE_T_H

#include <QVector4D> 
#include <QVector2D>
#include <QOpenGLWidget>
#include <QWheelEvent>
class QHyoda;

class QHyodaIceT : public QOpenGLWidget{
  Q_OBJECT
public:
  QHyodaIceT(QWidget*);
  ~QHyodaIceT();
public: 
  void sxyz(double*); 
  void setPov(QVector4D); 
  void setImage(QImage*);
  void saveGLState();
  void restoreGLState();
protected: 
  void initializeGL() Q_DECL_OVERRIDE;
  void resizeGL(int,int) Q_DECL_OVERRIDE;
  void paintGL() Q_DECL_OVERRIDE;
protected:
  void mousePressEvent(QMouseEvent *) Q_DECL_OVERRIDE;
  void mouseMoveEvent(QMouseEvent *) Q_DECL_OVERRIDE;
  void wheelEvent(QWheelEvent *) Q_DECL_OVERRIDE;
private:
  QWidget *parent;
  QVector2D mouse;
  QVector4D pov;
  QImage *image;
  GLuint texture;
};

#endif // Q_HYODA_ICE_T_H
