using System;

namespace SimdGenerator
{
  public interface ISimdClass
  {
    SimdType SimdType { get; }
    bool IsEmulated { get; }
    string SimdName { get; }
    string ClassName { get;}
    string FromScalar(string arg);
    string OpName(BinaryOperation op);
    //! Nom de l'instruction SIMD correspondante. Null si pas de SIMD correspondant.
    string OpName(UnaryOperation op);
    int DoubleVectorLength {get;}
    int Int32IndexLength{ get;}
    int NbNativeVector { get; }
  }
}

