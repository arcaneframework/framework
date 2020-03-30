#ifndef IFPSOLVERPROPERTY_H_
#define IFPSOLVERPROPERTY_H_

class IFPSolverProperty
{
 public:
  IFPSolverProperty() {}
  virtual ~IFPSolverProperty() {}
  typedef enum { Diag, ILU0, AMG, CprAmg } ePrecondType;

  typedef enum { All, Pressure } ePrecondEquationType;

  typedef enum {
    Normal,
    BlockJacobi,
    Optimized,
  } eIluAlgo;
};

#endif /*IFPSOLVERPROPERTY_H_*/
