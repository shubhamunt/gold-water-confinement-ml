# ============================================
# PARSE XDATCAR-MERGED TO COMBINED XYZ
# ============================================

def parse_xdatcar_to_xyz(filename):

    output_file = "all_frames.xyz"

    # 'open' function reads a file
    function = open(filename, "r")
    lines = function.readlines()
    function.close()

    # get scaling factor value
    scaling_factor = float(lines[1].strip())
    print("Scaling factor =", scaling_factor)

    # read lattice vectors
    a_line = lines[2].split()
    b_line = lines[3].split()
    c_line = lines[4].split()

    # assign values by converting to floats and apply scaling
    a = []  # empty matrix
    for value in a_line:
        a.append(float(value) * scaling_factor)

    b = []
    for value in b_line:
        b.append(float(value) * scaling_factor)

    c = []
    for value in c_line:
        c.append(float(value) * scaling_factor)

    # combine all lattice values
    lattice = []

    for value in a:
        lattice.append(value)

    for value in b:
        lattice.append(value)

    for value in c:
        lattice.append(value)

    lattice_str = ""
    for i in range(len(lattice)):
        lattice_str = lattice_str + str(lattice[i])
        if i != len(lattice) - 1:
            lattice_str = lattice_str + " "

    # final output
    header_line = 'Lattice="' + lattice_str + '" Properties=species:S:1:pos:R:3 pbc="T T T"'

    # Extract and count elements
    element_names = lines[5].split()
    count_line = lines[6].split()

    # total number of atoms
    num_of_atoms = 0
    for i in count_line:
        num_of_atoms = num_of_atoms + int(i)

    # convert counts to integers
    element_counts = []
    for value in count_line:
        element_counts.append(int(value))

    # make full element list
    elements = []
    for i in range(len(element_names)):
        for j in range(element_counts[i]):
            elements.append(element_names[i])

    # find all "Direct configuration" lines
    start_indices = []
    for i in range(len(lines)):
        if "Direct configuration" in lines[i]:
            start_indices.append(i)

    print("Number of frames found:", len(start_indices))

    # make combined xyz file
    out = open(output_file, "w")

    # Then do it for all the frames
    # Combine these XYZs
    for frame in range(len(start_indices)):

        start_index = start_indices[frame]

        # coordinates start from next line
        coordinate_start = start_index + 1
        coordinate_end = coordinate_start + num_of_atoms

        # extract those lines
        coords = []
        for i in range(coordinate_start, coordinate_end):
            coords.append(lines[i].strip())

        # convert Direct to Cartesian and make xyz lines
        xyz_line = []

        for i in range(num_of_atoms):
            parts = coords[i].split()

            dx = float(parts[0])
            dy = float(parts[1])
            dz = float(parts[2])

            x = dx * a[0] + dy * b[0] + dz * c[0]
            y = dx * a[1] + dy * b[1] + dz * c[1]
            z = dx * a[2] + dy * b[2] + dz * c[2]

            line = "{:<3s} {:15.8f} {:15.8f} {:15.8f}".format(elements[i], x, y, z)
            xyz_line.append(line)

        # write frame
        out.write(str(num_of_atoms) + "\n")
        out.write(header_line + "\n")
        for line in xyz_line:
            out.write(line + "\n")

        print("Wrote frame", frame + 1)

    out.close()
    print("Done. File written:", output_file)


# ============================================
# RADIAL DISTRIBUTION FUNCTION
# ============================================

import numpy as np
import matplotlib
matplotlib.use("Agg")   # save plots to files even on remote terminals
import matplotlib.pyplot as plt
import re


# ---------------------------------
# Distance under slab periodicity
#
# For two atoms i and j:
#
# r_ij = sqrt(dx^2 + dy^2 + dz^2)
#
# with minimum-image convention in x and y:
#
# dx = min(|x_i - x_j|, A - |x_i - x_j|)
# dy = min(|y_i - y_j|, B - |y_i - y_j|)
# dz = |z_i - z_j|
# ---------------------------------
def distance(a, b, A, B):
    dx = abs(a[0] - b[0])
    x = min(dx, abs(A - dx))

    dy = abs(a[1] - b[1])
    y = min(dy, abs(B - dy))

    dz = abs(a[2] - b[2])
    z = dz

    return np.sqrt(x**2 + y**2 + z**2)


# ---------------------------------
# Accessible sphere volume inside slab
#
# Full sphere volume:
# V_sphere(r) = (4/3) * pi * r^3
#
# If the sphere crosses the lower plane z_below,
# subtract the lower spherical cap:
#
# V_cap = pi * h^2 * (r - h/3)
#
# Final accessible volume:
#
# V_accessible(r) = V_sphere(r) - V_cap,below - V_cap,above
# ---------------------------------
def volume(z_center, r, z_below, z_above):
    v_total = 4.0 / 3.0 * np.pi * r**3

    v_below = 0.0
    if z_center - r < z_below:
        h_below = z_below - (z_center - r)
        v_below = np.pi * h_below**2 * (r - h_below / 3.0)

    v_above = 0.0
    if z_center + r > z_above:
        h_above = (z_center + r) - z_above
        v_above = np.pi * h_above**2 * (r - h_above / 3.0)

    v_accessible = v_total - v_below - v_above

    if v_accessible < 0.0:
        v_accessible = 0.0

    return v_accessible

class RDFTrajectory:   # class object
    def __init__(self, filename="all_frames.xyz", skip=1, z_cut=13.0, resolution=200):  # initial states -- parameters
        # object attributes
        self.filename = filename
        self.skip = int(skip)
        self.z_cut = float(z_cut)
        self.resolution = int(resolution)

        self.radii = None
        self.g_of_r = None

        self._read_xyz()   # read XYZ trajectory when the class is called

    # METHODS (functions inside a class)

    def _read_lattice_from_header(self, header_line):
        """
        Read lattice from line like:

        Lattice="10.218 0.0 0.0 0.0 8.849 0.0 0.0 0.0 34.450001" ...

        For orthorhombic box:
            A = a_x
            B = b_y
            C = c_z
        """
        match = re.search(r'Lattice="([^"]+)"', header_line)
        if match is None:
            raise ValueError("Lattice information not found in XYZ header.")

        lattice_values = [float(x) for x in match.group(1).split()]

        # lattice order:
        # a_x a_y a_z b_x b_y b_z c_x c_y c_z
        self.A = lattice_values[0]
        self.B = lattice_values[4]
        self.C = lattice_values[8]

    def _read_xyz(self):
        with open(self.filename, "r") as f:
            data = f.readlines()

        self.n_atoms = int(data[0].split()[0])
        block_size = self.n_atoms + 2

        # read lattice directly from XYZ header
        self._read_lattice_from_header(data[1])

        self.n_steps_total = len(data) // block_size
        self.n_steps = self.n_steps_total // self.skip

        # atom list from first frame
        self.atom_list = []
        for line in data[2:self.n_atoms + 2]:
            self.atom_list.append(line.split()[0])

        self.coordinates = np.zeros((self.n_steps, self.n_atoms, 3), dtype=float)

        saved_step = 0
        for step in range(0, self.n_steps_total, self.skip):
            if saved_step >= self.n_steps:
                break

            start = step * block_size + 2
            end = start + self.n_atoms

            for j, line in enumerate(data[start:end]):
                parts = line.split()
                self.coordinates[saved_step, j, 0] = float(parts[1])
                self.coordinates[saved_step, j, 1] = float(parts[2])
                self.coordinates[saved_step, j, 2] = float(parts[3])

            saved_step += 1

        print("Number of atoms =", self.n_atoms)
        print("Total number of frames =", self.n_steps_total)
        print("Frames used =", self.n_steps)
        print("A =", self.A)
        print("B =", self.B)
        print("C =", self.C)
        print("Unique atom types =", sorted(set(self.atom_list)))

    def _select_atoms(self, step, atom_type):
        # CONDITION:
        # Select water-region atoms (O, H) that sit ABOVE z_cut
        # i.e. z > z_cut
        selected = []

        for i, atom in enumerate(self.coordinates[step]):
            if self.atom_list[i] == atom_type and atom[2] > self.z_cut:
                selected.append(atom)

        return np.array(selected, dtype=float)

    def _select_surface_atoms(self, step, atom_type):
        # CONDITION:
        # Select surface atoms (Au) that sit AT or BELOW z_cut
        # i.e. z <= z_cut
        selected = []

        for i, atom in enumerate(self.coordinates[step]):
            if self.atom_list[i] == atom_type and atom[2] <= self.z_cut:
                selected.append(atom)

        return np.array(selected, dtype=float)

    def _select_by_type(self, step, atom_type):
        # CONDITION:
        # Au belongs to slab region below z_cut
        # O and H belong to water region above z_cut
        if atom_type == "Au":
            return self._select_surface_atoms(step, atom_type)
        else:
            return self._select_atoms(step, atom_type)

    def compute_number_density(self, atom_type):
        """
        rho_B = <N_B> / V_region

        CONDITIONS:
        - For water-region atoms (O, H):
              z_cut < z <= C
              V_region = A * B * (C - z_cut)

        - For surface atoms (Au):
              0 <= z <= z_cut
              V_region = A * B * z_cut
        """
        count = 0

        for step in range(self.n_steps):
            atoms = self._select_by_type(step, atom_type)
            count += len(atoms)

        if atom_type == "Au":
            volume_region = self.A * self.B * self.z_cut
        else:
            volume_region = self.A * self.B * (self.C - self.z_cut)

        avg_count = count / self.n_steps
        rho = avg_count / volume_region

        return rho

    def _get_r_cutoff(self, atom_type_1, atom_type_2):
        """
        CONDITIONS FOR X-AXIS / CUTOFF:

        1. For liquid-like RDFs:
              O-O, H-H, O-H, H-O
           use radial cutoff
              r_cutoff = min(A, B) / 2

        2. For Au-O and Au-H:
           use pair-distance axis from
              0 to C - z_cut

           This matches the requested slab-normal extent.

           NOTE:
           beyond min(A,B)/2, this is no longer a rigorously standard
           laterally-unbiased slab RDF normalization, but it is still a
           true pair-distance histogram/normalization over r.
        """
        au_pair = ((atom_type_1 == "Au" and atom_type_2 in ["O", "H"]) or
                   (atom_type_2 == "Au" and atom_type_1 in ["O", "H"]))

        if au_pair:
            return self.C - self.z_cut
        else:
            return min(self.A, self.B) / 2.0

    def compute_radial_distribution(self, atom_type_1, atom_type_2):
        """
        g_{A-B}(r) = n_{A-B}(r) / [ N_A * rho_B * DeltaV(r) ]

        where

            n_{A-B}(r) = number of A-B pairs in shell [r, r+dr]
            N_A        = average number of reference atoms A per frame
            rho_B      = number density of atom type B
            DeltaV(r)  = accessible shell volume

        IMPORTANT:
        The histogram is essential.

            counts[i] = n(r_i)

        CONDITIONS:
        - O-O, H-H, O-H: use radial cutoff = min(A,B)/2
        - O-Au, H-Au: use radial cutoff = C - z_cut
        """

        self.r_cutoff = self._get_r_cutoff(atom_type_1, atom_type_2)
        dr = self.r_cutoff / self.resolution

        # radii at left edge of each bin: 0 <= r < r_cutoff
        self.radii = np.linspace(0.0, self.r_cutoff, self.resolution, endpoint=False)

        # HISTOGRAM OF PAIR DISTANCES
        counts = np.zeros(self.resolution, dtype=float)

        rho = self.compute_number_density(atom_type_2)
        n_ref_total = 0
        same_species = (atom_type_1 == atom_type_2)

        atoms1_per_step = [self._select_by_type(s, atom_type_1) for s in range(self.n_steps)]
        atoms2_per_step = [self._select_by_type(s, atom_type_2) for s in range(self.n_steps)]

        # ---------------------------------
        # Step 1: build histogram n(r)
        # ---------------------------------
        for step in range(self.n_steps):
            print("Step", step + 1, "of", self.n_steps, "for", atom_type_1, "-", atom_type_2)

            atoms1 = atoms1_per_step[step]
            atoms2 = atoms2_per_step[step]

            print("Selected:", atom_type_1, len(atoms1), "|", atom_type_2, len(atoms2))

            if len(atoms1) == 0 or len(atoms2) == 0:
                continue

            n_ref_total += len(atoms1)

            if same_species:
                # CONDITION:
                # for same-species RDF, use only unique pairs j > i
                # then add 2 because i sees j and j sees i
                for i in range(len(atoms1)):
                    for j in range(i + 1, len(atoms2)):
                        r = distance(atoms1[i], atoms2[j], self.A, self.B)

                        if r < self.r_cutoff:
                            idx = int(r / dr)
                            counts[idx] += 2.0
            else:
                # CONDITION:
                # for unlike pairs, use all A-B combinations
                for i in range(len(atoms1)):
                    for j in range(len(atoms2)):
                        r = distance(atoms1[i], atoms2[j], self.A, self.B)

                        if r < self.r_cutoff:
                            idx = int(r / dr)
                            counts[idx] += 1.0

        self.g_of_r = np.zeros(self.resolution, dtype=float)

        if n_ref_total == 0 or rho == 0.0:
            print("No RDF computed for", atom_type_1, atom_type_2)
            return self.radii, self.g_of_r

        avg_n_ref = n_ref_total / self.n_steps

        # ---------------------------------
        # Step 2: normalize histogram
        # ---------------------------------
        for i in range(self.resolution):
            r1 = self.radii[i]
            r2 = r1 + dr

            shell_volume_total = 0.0
            ref_atoms_for_volume = 0

            for step in range(self.n_steps):
                for atom in atoms1_per_step[step]:
                    # CONDITION:
                    # accessible shell volume is computed using z-truncated sphere
                    v1 = volume(atom[2], r1, self.z_cut, self.C)
                    v2 = volume(atom[2], r2, self.z_cut, self.C)
                    shell_volume_total += (v2 - v1)
                    ref_atoms_for_volume += 1

            if ref_atoms_for_volume == 0:
                continue

            shell_volume = shell_volume_total / ref_atoms_for_volume
            ideal_count = avg_n_ref * rho * shell_volume * self.n_steps

            if ideal_count > 0.0:
                self.g_of_r[i] = counts[i] / ideal_count

        print("Finished RDF for", atom_type_1, atom_type_2)
        return self.radii, self.g_of_r

    def _plot_rdf(self, atom_type_1, atom_type_2, filename_base):
        print("Computing RDF for", atom_type_1, "-", atom_type_2)

        r, g = self.compute_radial_distribution(atom_type_1, atom_type_2)

        png_name = filename_base + ".png"
        dat_name = filename_base + ".dat"

        # always write numeric data
        with open(dat_name, "w") as f:
            f.write("# r(Ang)    g(r)\n")
            for ri, gi in zip(r, g):
                f.write(f"{ri:15.8f} {gi:15.8f}\n")

        plt.figure(figsize=(4, 4))
        plt.plot(r, g, linewidth=2)
        plt.xlabel("r (Å)")
        plt.ylabel(f"g$_{{{atom_type_1}-{atom_type_2}}}$(r)")

        # CONDITION:
        # x-axis for Au-O and Au-H is now 0 to C - z_cut
        # x-axis for O-O, H-H, O-H is 0 to min(A,B)/2
        plt.xlim(0.0, self.r_cutoff)

        plt.tight_layout()
        plt.savefig(png_name, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", png_name)
        print("Saved:", dat_name)

    # ---------------------------------
    # RDFs in pair distance r
    # ---------------------------------
    def rdf_OO(self):
        # CONDITION: O-O is a radial RDF in r
        self._plot_rdf("O", "O", "rdf_OO")

    def rdf_HH(self):
        # CONDITION: H-H is a radial RDF in r
        self._plot_rdf("H", "H", "rdf_HH")

    def rdf_OH(self):
        # CONDITION: O-H is a radial RDF in r
        self._plot_rdf("O", "H", "rdf_OH")

    def rdf_OAu(self):
        # CONDITION: true Au-O RDF in r, but x-axis runs 0 to C - z_cut
        self._plot_rdf("O", "Au", "rdf_OAu")

    def rdf_HAu(self):
        # CONDITION: true Au-H RDF in r, but x-axis runs 0 to C - z_cut
        self._plot_rdf("H", "Au", "rdf_HAu")


# ============================================
# RUN
# ============================================

#filename = "XDATCAR-merged"
#parse_xdatcar_to_xyz(filename)

traj = RDFTrajectory(filename="all_frames.xyz", skip=1, z_cut=13.0, resolution=40)
traj.rdf_OO()
traj.rdf_HH()
traj.rdf_OAu()
traj.rdf_HAu()
traj.rdf_OH()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from scipy.signal import fftconvolve


def distance(p1, p2, A, B):
    """
    Minimum-image distance in x and y, direct in z.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    if dx > A / 2.0:
        dx -= A
    elif dx < -A / 2.0:
        dx += A

    if dy > B / 2.0:
        dy -= B
    elif dy < -B / 2.0:
        dy += B

    return np.sqrt(dx * dx + dy * dy + dz * dz)


class VDOSWaterOH:
    def __init__(self, filename="all_frames.xyz", timestep_fs=1.0):
        self.filename = filename
        self.timestep_fs = float(timestep_fs)
        self.dt_s = self.timestep_fs * 1.0e-15

        self.positions = None
        self.symbols = None
        self.A = None
        self.B = None
        self.n_frames = None
        self.n_atoms = None

        self.oh_pairs = None
        self.velocities = None
        self.conv_matrix = None
        self.vacf = None
        self.wavenumbers = None
        self.vdos = None

        self._read_xyz()

    def _read_lattice(self, header_line):
        match = re.search(r'Lattice="([^"]+)"', header_line)
        if match is None:
            raise ValueError("Could not find Lattice=\"...\" in XYZ header.")
        values = [float(x) for x in match.group(1).split()]
        self.A = values[0]
        self.B = values[4]

    def _read_xyz(self):
        with open(self.filename, "r") as f:
            raw = f.readlines()

        n_atoms = int(raw[0].split()[0])
        block_size = n_atoms + 2
        n_frames = len(raw) // block_size

        self._read_lattice(raw[1])

        symbols = [raw[2 + i].split()[0] for i in range(n_atoms)]
        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)

        for fidx in range(n_frames):
            start = fidx * block_size + 2
            for j in range(n_atoms):
                parts = raw[start + j].split()
                positions[fidx, j, 0] = float(parts[1])
                positions[fidx, j, 1] = float(parts[2])
                positions[fidx, j, 2] = float(parts[3])

        self.symbols = symbols
        self.positions = positions
        self.n_frames = n_frames
        self.n_atoms = n_atoms

        print(f"Frames loaded    = {n_frames}")
        print(f"Atoms per frame  = {n_atoms}")
        print(f"Cell: A = {self.A:.4f} Å, B = {self.B:.4f} Å")
        print(f"dt               = {self.timestep_fs} fs")
        print(f"Unique elements  = {sorted(set(symbols))}")

    def identify_oh_pairs(self, reference_frame=0, oh_cutoff=1.3):
        """
        Assign up to 2 nearest H atoms within oh_cutoff to each O.
        """
        pos = self.positions[reference_frame]
        o_indices = [i for i, s in enumerate(self.symbols) if s == "O"]
        h_indices = [i for i, s in enumerate(self.symbols) if s == "H"]

        oh_pairs = []

        for o_idx in o_indices:
            neighbours = []
            for h_idx in h_indices:
                r = distance(pos[o_idx], pos[h_idx], self.A, self.B)
                if r < oh_cutoff:
                    neighbours.append((r, h_idx))

            neighbours.sort(key=lambda x: x[0])

            for r, h_idx in neighbours[:2]:
                oh_pairs.append((o_idx, h_idx))

        self.oh_pairs = oh_pairs

        print(f"O atoms          = {len(o_indices)}")
        print(f"H atoms          = {len(h_indices)}")
        print(f"O-H pairs found  = {len(self.oh_pairs)}")

        if len(self.oh_pairs) < 2 * len(o_indices):
            print("WARNING: Some O atoms have fewer than 2 H within the cutoff.")
            print("Try increasing oh_cutoff, e.g. to 1.4 or 1.5 Å if needed.")

        return self.oh_pairs

    def write_oh_pairs(self, reference_frame=0, outfile="oh_pairs.txt", oh_cutoff=1.3):
        """
        Write total O-H pairs and coordinates of O/H atoms to a text file.
        """
        self.identify_oh_pairs(reference_frame=reference_frame, oh_cutoff=oh_cutoff)

        pos = self.positions[reference_frame]
        n_o = sum(1 for s in self.symbols if s == "O")
        n_h = sum(1 for s in self.symbols if s == "H")

        with open(outfile, "w") as f:
            f.write("O-H PAIRS INFORMATION\n")
            f.write("=====================\n\n")
            f.write(f"Reference frame   : {reference_frame}\n")
            f.write(f"Total O atoms     : {n_o}\n")
            f.write(f"Total H atoms     : {n_h}\n")
            f.write(f"Total O-H pairs   : {len(self.oh_pairs)}\n")
            f.write(f"O-H cutoff (Å)    : {oh_cutoff}\n\n")

            f.write(
                "Pair    O_index         O_x         O_y         O_z"
                "      H_index         H_x         H_y         H_z    O-H_dist(Å)\n"
            )
            f.write("-" * 110 + "\n")

            for pair_id, (o_idx, h_idx) in enumerate(self.oh_pairs, start=1):
                O = pos[o_idx]
                H = pos[h_idx]
                dist = distance(O, H, self.A, self.B)

                f.write(
                    f"{pair_id:4d}  "
                    f"{o_idx:7d}  {O[0]:11.5f} {O[1]:11.5f} {O[2]:11.5f}  "
                    f"{h_idx:7d}  {H[0]:11.5f} {H[1]:11.5f} {H[2]:11.5f}  "
                    f"{dist:10.5f}\n"
                )

        print(f"Saved O-H pair details to: {outfile}")
        print(f"Total O-H pairs = {len(self.oh_pairs)}")

    def compute_velocities(self):
        """
        Central-difference velocities for all atoms.
        """
        R = self.positions
        dR = R[2:] - R[:-2]

        dR[:, :, 0] = np.where(dR[:, :, 0] > self.A / 2.0, dR[:, :, 0] - self.A, dR[:, :, 0])
        dR[:, :, 0] = np.where(dR[:, :, 0] < -self.A / 2.0, dR[:, :, 0] + self.A, dR[:, :, 0])

        dR[:, :, 1] = np.where(dR[:, :, 1] > self.B / 2.0, dR[:, :, 1] - self.B, dR[:, :, 1])
        dR[:, :, 1] = np.where(dR[:, :, 1] < -self.B / 2.0, dR[:, :, 1] + self.B, dR[:, :, 1])

        self.velocities = dR * 1.0e-10 / (2.0 * self.dt_s)

        print(f"Velocities shape = {self.velocities.shape}")
        return self.velocities

    def compute_convergence_matrix(self):
        """
        Projected O-H relative velocity for each identified O-H pair.
        """
        if self.oh_pairs is None:
            self.identify_oh_pairs()
        if self.velocities is None:
            self.compute_velocities()

        R_mid = self.positions[1:-1]
        n_steps = self.velocities.shape[0]
        n_pairs = len(self.oh_pairs)

        M = np.zeros((n_pairs, n_steps), dtype=np.float64)

        for b, (o_idx, h_idx) in enumerate(self.oh_pairs):
            dr = R_mid[:, h_idx, :] - R_mid[:, o_idx, :]

            dr[:, 0] = np.where(dr[:, 0] > self.A / 2.0, dr[:, 0] - self.A, dr[:, 0])
            dr[:, 0] = np.where(dr[:, 0] < -self.A / 2.0, dr[:, 0] + self.A, dr[:, 0])

            dr[:, 1] = np.where(dr[:, 1] > self.B / 2.0, dr[:, 1] - self.B, dr[:, 1])
            dr[:, 1] = np.where(dr[:, 1] < -self.B / 2.0, dr[:, 1] + self.B, dr[:, 1])

            bond_len = np.linalg.norm(dr, axis=1)
            safe_len = np.where(bond_len > 1e-12, bond_len, 1.0)
            e_oh = dr / safe_len[:, np.newaxis]

            dv = self.velocities[:, h_idx, :] - self.velocities[:, o_idx, :]
            M[b, :] = np.sum(dv * e_oh, axis=1)

        self.conv_matrix = M

        print(f"Projected velocity matrix shape = {M.shape}")
        return M

    def compute_vacf(self):
        """
        VACF using scipy.signal.fftconvolve.
        """
        if self.conv_matrix is None:
            self.compute_convergence_matrix()

        n_pairs, n_steps = self.conv_matrix.shape
        all_acf = np.zeros((n_pairs, n_steps), dtype=np.float64)

        for b in range(n_pairs):
            x = self.conv_matrix[b]
            acf_full = fftconvolve(x, x[::-1], mode="full")
            acf = acf_full[n_steps - 1:]
            acf /= np.arange(n_steps, 0, -1, dtype=np.float64)
            all_acf[b] = acf

        vacf = np.mean(all_acf, axis=0)

        if abs(vacf[0]) > 1e-20:
            vacf /= vacf[0]

        self.vacf = vacf

        print(f"VACF length = {len(vacf)}")
        return self.vacf

    def compute_vdos(self):
        """
        FFT of VACF -> VDOS.
        """
        if self.vacf is None:
            self.compute_vacf()

        N = len(self.vacf)
        c_cm_s = 2.99792458e10

        window = np.hanning(N)
        vacf_windowed = self.vacf * window

        fft_vals = np.fft.rfft(vacf_windowed)
        freqs_hz = np.fft.rfftfreq(N, d=self.dt_s)

        self.wavenumbers = freqs_hz / c_cm_s
        self.vdos = np.abs(fft_vals)
        self.vdos[0] = 0.0

        print("VDOS computed.")
        return self.wavenumbers, self.vdos

    def plot_vacf(self, outfile="vacf_water_oh.png", max_lag_fs=200.0):
        if self.vacf is None:
            self.compute_vacf()

        n_plot = min(int(max_lag_fs / self.timestep_fs), len(self.vacf))
        lag_fs = np.arange(n_plot) * self.timestep_fs

        plt.figure(figsize=(8, 4))
        plt.plot(lag_fs, self.vacf[:n_plot], linewidth=1.5)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        plt.xlabel("Time lag (fs)")
        plt.ylabel("VACF(t) / VACF(0)")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        np.savetxt(
            "vacf_water_oh.dat",
            np.column_stack((lag_fs, self.vacf[:n_plot])),
            header="time_fs   vacf_normalized"
        )

        print(f"Saved VACF plot: {outfile}")
        print("Saved VACF data: vacf_water_oh.dat")

    def plot_vdos(self, outfile="vdos_water_oh.png", wn_min=2500.0, wn_max=4000.0):
        if self.vdos is None:
            self.compute_vdos()

        plt.figure(figsize=(8, 4))
        plt.plot(self.wavenumbers, self.vdos, linewidth=1.5)
        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.ylabel("VDOS (arb. units)")
        plt.xlim(wn_min, wn_max)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

        np.savetxt(
            "vdos_water_oh.dat",
            np.column_stack((self.wavenumbers, self.vdos)),
            header="wavenumber_cm^-1   vdos"
        )

        print(f"Saved VDOS plot: {outfile}")
        print("Saved VDOS data: vdos_water_oh.dat")


# ============================================================
# RUN
# ============================================================
vdos = VDOSWaterOH(
    filename="all_frames.xyz",
    timestep_fs=1.0
)

# 1. Identify O-H pairs and write coordinates to text file
vdos.write_oh_pairs(
    reference_frame=0,
    outfile="oh_pairs.txt",
    oh_cutoff=1.3
)

# 2. Compute projected O-H velocities
vdos.compute_velocities()
vdos.compute_convergence_matrix()

# 3. Compute and plot VACF
vdos.compute_vacf()
vdos.plot_vacf(
    outfile="vacf_water_oh.png",
    max_lag_fs=200.0
)

# 4. Compute and plot VDOS
vdos.compute_vdos()
vdos.plot_vdos(
    outfile="vdos_water_oh.png",
    wn_min=0.0,
    wn_max=4000.0
)

