#ifndef Q_HYODA_X11_EMBED_H
#define Q_HYODA_X11_EMBED_H

#include <QtWidgets>
//#include <QX11EmbedContainer>
class QHyodaX11;

class QHyodaX11Embed: public QWidget{
  Q_OBJECT
public:
  QHyodaX11Embed(QWidget*, QHyodaX11*);
  ~QHyodaX11Embed(void);
};

#endif // Q_HYODA_X11_EMBED_H
