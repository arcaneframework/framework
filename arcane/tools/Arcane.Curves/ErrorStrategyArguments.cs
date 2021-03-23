using System;
namespace Arcane.Curves
{
  //! Arguments pour la comparaison de courbe.
  public class ErrorStrategyArguments
  {
    //! Valeur minimale de l'abscisse à prendre en compte
    public double MinX { get; set; }
    //! Valeur maximale de l'abscisse à prendre en compte
    public double MaxX { get; set; }

    public ErrorStrategyArguments ()
    {
      MinX = double.MinValue;
      MaxX = double.MaxValue;
    }
  }
}
