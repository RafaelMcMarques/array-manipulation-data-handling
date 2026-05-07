from pattern import Checker, Circle, Spectrum

circle = Circle(1000, 300, (0, 0))
circle.draw()
circle.show()

checker = Checker(1000, 10)
checker.draw()
checker.show()

spectrum = Spectrum(1000)
spectrum.draw()
spectrum.show()
