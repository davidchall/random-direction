from math import sqrt
import abc

import numpy as np
import pint
ureg = pint.UnitRegistry()


class BeamBase(metaclass=abc.ABCMeta):

    def __init__(self, cutoff_theta_x, cutoff_theta_y):
        self.cutoff_theta_x = cutoff_theta_x
        self.cutoff_theta_y = cutoff_theta_y
        self.n_attempt = 0
        self.n_success = 0

    @property
    def cutoff_theta_x(self):
        return self._cutoff_theta_x

    @cutoff_theta_x.setter
    def cutoff_theta_x(self, cutoff_theta_x):
        assert(cutoff_theta_x <= np.pi * ureg.radian)
        self._cutoff_theta_x = cutoff_theta_x
        self._cutoff_x1 = self._angle_to_marsaglia_coordinate(cutoff_theta_x)

    @property
    def cutoff_theta_y(self):
        return self._cutoff_theta_y

    @cutoff_theta_y.setter
    def cutoff_theta_y(self, cutoff_theta_y):
        assert(cutoff_theta_y <= np.pi * ureg.radian)
        self._cutoff_theta_y = cutoff_theta_y
        self._cutoff_x2 = self._angle_to_marsaglia_coordinate(cutoff_theta_y)

    @staticmethod
    def _angle_to_marsaglia_coordinate(theta):
        y = np.tan(theta)**2
        if theta <= np.pi * ureg.rad / 2.:
            return sqrt(y/(y+1) - 1./sqrt(y+1) + 1./(y+1)) / sqrt(2)
        else:
            return sqrt(y/(y+1) + 1./sqrt(y+1) + 1./(y+1)) / sqrt(2)

    @staticmethod
    def _marsaglia_transform(x1, x2):
        x = 2 * x1 * np.sqrt(1 - np.square(x1) - np.square(x2))
        y = 2 * x2 * np.sqrt(1 - np.square(x1) - np.square(x2))
        z = 1 - 2 * (np.square(x1) + np.square(x2))

        return x, y, z

    @abc.abstractmethod
    def _generate_marsaglia(self, size):
        pass

    def generate(self, size):
        self.n_success += size

        x1, x2 = self._generate_marsaglia(size)
        return self._marsaglia_transform(x1, x2)

    def efficiency(self):
        return self.n_success / self.n_attempt if self.n_attempt else None


class UniformBeam(BeamBase):

    def __init__(self, cutoff_theta_x, cutoff_theta_y):
        super().__init__(cutoff_theta_x, cutoff_theta_y)

    def _generate_marsaglia(self, size):
        self.n_attempt += size

        x1 = np.random.uniform(-self._cutoff_x1, self._cutoff_x1, size)
        x2 = np.random.uniform(-self._cutoff_x2, self._cutoff_x2, size)

        valid = (np.square(x1)/self._cutoff_x1**2) + (np.square(x2)/self._cutoff_x2**2) <= 1.
        size_regenerate = np.size(valid) - np.count_nonzero(valid)
        if size_regenerate > 0:
            x1[~valid], x2[~valid] = self._generate_marsaglia(size_regenerate)

        return x1, x2


class GaussianBeam(BeamBase):

    def __init__(self, spread_theta_x, spread_theta_y,
                 cutoff_theta_x=np.pi*ureg.radian, cutoff_theta_y=np.pi*ureg.radian):

        super().__init__(cutoff_theta_x, cutoff_theta_y)
        self.spread_theta_x = spread_theta_x
        self.spread_theta_y = spread_theta_y

    @property
    def spread_theta_x(self):
        return self._spread_theta_x

    @spread_theta_x.setter
    def spread_theta_x(self, spread_theta_x):
        assert(spread_theta_x <= np.pi * ureg.radian)
        self._spread_theta_x = spread_theta_x
        self._spread_x1 = self._angle_to_marsaglia_coordinate(spread_theta_x)

    @property
    def spread_theta_y(self):
        return self._spread_theta_y

    @spread_theta_y.setter
    def spread_theta_y(self, spread_theta_y):
        assert(spread_theta_y <= np.pi * ureg.radian)
        self._spread_theta_y = spread_theta_y
        self._spread_x2 = self._angle_to_marsaglia_coordinate(spread_theta_y)

    def _generate_marsaglia(self, size):
        self.n_attempt += size

        x1 = np.random.normal(0, self._spread_x1, size)
        x2 = np.random.normal(0, self._spread_x2, size)

        valid = (np.square(x1)/self._cutoff_x1**2) + (np.square(x2)/self._cutoff_x2**2) <= 1.
        size_regenerate = np.size(valid) - np.count_nonzero(valid)
        if size_regenerate > 0:
            x1[~valid], x2[~valid] = self._generate_marsaglia(size_regenerate)

        return x1, x2
