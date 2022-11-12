"""
Code create to analyse CFD experimental data tailored for CWE application.
@author: Abiy
"""
import numpy as np
from scipy import stats
import pandas as pd
from scipy import signal

class CAARC:
    """
    Contains the geometry of CAARC building.
    """
    #Constants defined in ESDU for evaluating mean wind speed profile
    H = 182.88
    B = 45.72
    D = 30.48

class ESDU:
    """
    Contains data and function used to calculate atmospheric boundary layer
    profile based on ESDU82* documents. All function that end up by _target are
    calculation based on the target site. 
    """
    #Constants defined in ESDU for evaluating mean wind speed profile
    __omega = 72.9e-6
    __A = -1
    __B = 6
    __a1 = 2*(np.log(__B) - __A) + 1/6
    __v_r = 0
    __z_r = 0
    __z_0 = 0   
    __phi = 0 
    __f = 0 
    __u_star = 0
    __h = 0
    __V_h = 0
    
    def __init__(self, v_r, z_r, z_0, phi = 45):       
        self.__v_r = v_r
        self.__z_r = z_r
        self.__z_0 = z_0
        self.__phi = phi
        self.__f = self.__get_f()  
        self.__u_star = self.__get_u_star()
        self.__z = self.__z_r
        self.__h = self.get_h()
        self.__V_h = self.__get_V_h()

    def __get_u_star(self):
        z0 = self.__z_0
        z = self.__z_r 
        v = self.__v_r
        f= self.__f
        return (v - 2.5*34.5*f*z)/(2.5*np.log(z/z0))
#        return (v)/(2.5*np.log(z/z0))

    def get_u_star(z_r, v_r, z0):
        return v_r/(2.5*np.log(z_r/z0))
        
    def get_ustar(self):
        return self.__u_star
        
    def get_h(self):
#        self.__h = 2500
#        self.__f = self.__u_star/(self.__B*self.__h)  
#        return self.__h
        return self.__u_star/(self.__B*self.__f)  

    def __get_f(self):
        return 2*self.__omega*np.sin(np.radians(self.__phi))

    def __get_V_h(self):
        return self.__u_star*2.5*(np.log(self.__u_star/(self.__f*self.__z_0))-self.__A)
    
    def get_V(self, Z, d=0):
        h = self.__h
        z0 = self.__z_0
        a1 = self.__a1
        u_star = self.__u_star
        V  = np.zeros(len(Z))
        for i in range(len(Z)):
            z = Z[i]
            V[i] = u_star*2.5*(np.log(z/z0) + a1*z/h + (1 - a1/2)*(z/h)**2 - (4/3)*(z/h)**3 + (1/4)*(z/h)**4)
#        return u_star*2.5*(np.log(z/z0) + a1*z/h + (1 - a1/2)*(z/h)**2 - (4/3)*(z/h)**3 + (1/4)*(z/h)**4)
#        return u_star*2.5*(np.log((z-d)/z0) + a1*(z-d)/h) 
        return V

    #Calculates the ABL profile given the coordinate and d
    def get_Uav(self, z, d=0.0):
        u_av = np.zeros(len(z))
        for i in range(len(z)):
            u_av[i] = self.get_V_z(z[i], d)
        return u_av
    
    #Calculates the ABL profile given the coordinate and d
    def get_Iu(self, z, d=0.0):
        I_u = np.zeros(len(z))
        for i in range(len(z)):
            I_u[i] = self.get_I(z[i])[0]
        return I_u

    def get_V_z(self, z, d=0):
        h = self.__h
        z0 = self.__z_0
        a1 = self.__a1
        u_star = self.__u_star
        return u_star*2.5*(np.log(z/z0) + a1*z/h + (1 - a1/2)*(z/h)**2 - (4/3)*(z/h)**3 + (1/4)*(z/h)**4)
#        return u_star*2.5*(np.log((z-d)/z0) + a1*(z-d)/h)

    def get_I(self, z):
        V_z = self.get_V_z(z)
        eta = 1 - self.__B*self.__f*z/self.__u_star
        p = eta**16
        std_u = self.__u_star*(7.5*eta*(0.538 + 0.09*np.log(z/self.__z_0))**p)/(1 + 0.156*np.log(self.__u_star/(self.__f*self.__z_0)))
        std_v = std_u*(1 - 0.22*(np.cos(np.pi*z/(2*self.__h)))**4)
        std_w = std_u*(1 - 0.45*(np.cos(np.pi*z/(2*self.__h)))**4) 
        I = np.zeros(3)
        I[0] = std_u/V_z 
        I[1] = std_v/V_z
        I[2] = std_w/V_z
        return I

    def get_uw_bar(self, z, rho=1.25):
        return rho*(self.__u_star)**2*(1 - z/self.__h)**2  
    
    def get_uw(self, z):
        return (self.__u_star)**2*(1 - z/self.__h)**2  
        
    def get_f_given_h(self, h, z_r, v_r, z0):
        u_star = self.get_u_star(z_r, v_r, z0)
        return u_star/(6*h) 
        
        
    def get_L(self, z):
        h = self.__h
        u_star = self.__u_star
        f = self.__f
        z_0 = self.__z_0     
        B =self.__B
        
        eta = 1 - B*f*z/u_star
        p = eta**16

        std_u = u_star*(7.5*eta*(0.538 + 0.09*np.log(z/z_0))**p)/(1 + 0.156*np.log(u_star/(f*z_0)))
        std_v = std_u*(1 - 0.22*(np.cos(np.pi*z/(2*h)))**4)
        std_w = std_u*(1 - 0.45*(np.cos(np.pi*z/(2*h)))**4) 
        
        A = 0.115*(1 + 0.315*(1-z/h)**6)**(2/3)  
        z_c = 0.39*h*(u_star/(f*z_0))**(-1/8)
        K_inf = 0.188
        K_z = K_inf
        
        if z < z_c:
            K_z = K_inf*np.sqrt(1 - (1 - z/z_c)**2)
            
        L = np.zeros(3)
        L[0] = ((A**(3/2))*((std_u/u_star)**3)*z)/(2.5*(K_z**(3/2))*((1-z/h)**2)*(1 + 5.75*z/h))
        
        L[1] = 0.5*L[0]*(std_v/std_u)**3
        L[2] = 0.5*L[0]*(std_w/std_u)**3   
        return L
        
    def get_Spectrum(self, f, Uav, I, L, z):
        h = self.__h
        A = 0.115*(1.0 + 0.315*(1.0-z/h)**6.0)**(2.0/3.0)
        a = 0.535 + 2.76*(0.138 - A)*0.68;  # alpha
        b1 = 2.357*a - 0.761   # beta1
        b2 = 1.0 - b1           # beta2
        
        S = np.zeros((3, len(f)))
        
        for i in range(len(f)):
            n = f[i]
            nu = n*L[0]/Uav
            F1 = 1+0.455*np.exp(-0.76*nu/a**-0.8)
            rSuu = b1*((2.987*nu/a)/(1 + (2*np.pi*nu/a)**2)**(5.0/6.0))+b2*((1.294*nu/a)/(1.0 +(np.pi*nu/a)**2.0)**(5.0/6.0))*F1
            varU = (I[0]*Uav)**2
            S[0,i] = rSuu*varU/n
        
            nv = n*L[1]/S.U
            F2v = 1+2.88*np.exp(-0.218*nv/a^-0.9)
            rSvv = b1*(((2.987*(1+(8.0/3.0)*(4*np.pi*nv/a)**2.0)))*(nv/a)/(1+(4*np.pi*nv/a)**2.0)**(11.0/6.0))+b2*((1.294*nv/a)/((1.0 +(2*np.pi*nv/a)**2.0)**(5.0/6.0)))*F2v
            varV = (I[1]*Uav)**2.0
            S[1,i] = rSvv*varV/n
            
            nw = n*L[2]/Uav
            F2w = 1 + 2.88*np.exp(-0.218*nw/(a**-0.9))
            rSww = b1*(((2.987*(1+(8.0/3.0)*(4*np.pi*nw/a)**2.0)))*(nw/a)/(1.0 + (4*np.pi*nw/a)**2.0)**(11.0/6.0)) + b2*((1.294*nw/a)/((1+(2*np.pi*nw/a)**2.0)**(5.0/6.0)))*F2w
            varW = (I[2]*Uav)**2.0
            S[2,i] = rSww*varW/n
        
        return S    

    def write_tabel(self):    
        print('phi = ', self.__phi) 
        print('V_r = ', self.__v_r)
        print('z_r = ', self.__z_r)
        print('f = ', self.__f)
        print('z_0 = ', self.__z_0)
        print('u_star = ', self.__u_star)
        print('h = ', self.__h)
        print('V_h = ', self.__V_h)
        print('A = ', self.__A)
        print('B = ', self.__B)
        return None    

def setup_plot(plt, font_size=20, legend_font_size=20, axis_font_size=20):
    fig = plt.figure(facecolor='white')
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : font_size}
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rc('font', **font)    
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('xtick', labelsize=axis_font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=axis_font_size)    # fontsize of the tick labels
    plt.rc('axes', linewidth=1.25)    
    plt.rc('legend', fontsize=legend_font_size)
    plt.rc('text', usetex=True)

    return plt, fig

#Creates probes for a building for a given dimensions
def create_probes_around_building(B, D, N_B, N_D, z):
    """
    Creates probes on face B(front or north) -> D(left or east) -> B(back or south). 
    A small increment('e') is added to make the probes always on the mesh side. 
    """
    e = B/100000.0    
    dy = B/N_B
    dx = D/N_D
    
    x = np.linspace(dx/3.0, D-dx/3.0, num=N_D,endpoint=True)
    y = np.linspace(dy/3.0, B-dy/3.0, num=N_B,endpoint=True)
    
    probes = np.zeros((len(x) + 2*len(y), 3))
    
    for i in range(len(y)):
        probes[i,0] = -e
        probes[i,1] = y[i]
        probes[len(y) + len(x) + i, 0] = D + e
        probes[len(y) + len(x) + i, 1] = B - y[i]
    
    for i in range(len(x)):
        probes[len(y) + i,0] = x[i]
        probes[len(y) + i,1] = B + e
    
    transform = np.zeros((1,3))
    transform[0,0] = -D/2.0
    transform[0,1] = -B/2.0
    transform[0,2] = z
    
    probes += transform

    return probes

#Creates velocity probes in a given direction
def create_velocity_probes(min_loc, max_loc, n_probes):
    """
    Creates n probes on a given line defined by min and max locations.
    """
    probes = np.zeros((n_probes,3)) 
    probes[:,:] = min_loc
    
    for i in range(n_probes):
        probes[i,:] = min_loc + i*(max_loc - min_loc)/n_probes   
    return probes
   

class profile: 


    def __init__(self, profile_type, z_0, scale, *file_name):
        self.file_name = file_name
        self.z_0 = z_0
        self.type = profile_type
        self.scale = scale
        self.dt = 0.0            #Time step
        self.f_max  = 0.0        #Maximum frequency 
        self.time = []           #Time sequence
        self.Nt = 0              #Number of time steps 
        self.Np = 0              #Number of measurement points 
        self.locations = []      #x, y, z coordinates of the points 
        self.z = []              #just the z coordinate of the points.
        self.U = []              #Velocity time history
        self.cmpnt  = ['u', 'v', 'w']
        self.start_time = 0.00
        self.end_time = None
        self.no_cmpnt = 3
        self.Uav = []
        self.I = []
        self.L = []
        self.uw_bar = []
        self.u = []        
        self.__read_file()
        self.__calculate_profiles()
        
    def __read_file(self):

        #Read the data based on the type of the data. 
        if self.type == 'experiment':
            self.__read_experemetal_data()
        elif self.type == 'inflow':
            self.__read_inflow_data()
        elif self.type == 'cfd':
            self.__read_cfd_data()
        else:
            print ('Data type is not recognized!')

    def __read_inflow_data(self):
        # Read wind tunnel measurements
        self.file_name = self.file_name[0]
        u = np.transpose(np.loadtxt(self.file_name +  '/Ux'))
        self.Np = np.shape(u)[0]
        self.Nt = np.shape(u)[1]
               
        self.U = np.zeros((self.Np, self.no_cmpnt , self.Nt))        
        self.U[:,0,:] = u
        self.U[:,1,:] = np.transpose(np.loadtxt(self.file_name +  '/Uy'))
        self.U[:,2,:] = np.transpose(np.loadtxt(self.file_name +  '/Uz'))
        self.locations = np.zeros((self.Np, self.no_cmpnt))
        
        try:
            self.z = np.loadtxt(self.file_name + '/sampleABL', usecols=0)
        except:
            self.z = np.loadtxt(self.file_name + '/nearestPoints', usecols=2)
            self.locations = np.loadtxt(self.file_name + '/nearestPoints')
            
        self.time = np.loadtxt(self.file_name + '/time')
        self.dt = np.mean(np.diff(self.time))
        self.f_max = 1.0/(2.0*self.dt)
        self.end_time = self.time[-1]
        del u

#        if(self.start_time != None):
#            start_index = int(np.argmax(self.time > self.start_time))
#            self.time = self.time[start_index:]
#            self.U = self.U[:,:,start_index:]
#
#        if(self.end_time != None):
#            end_index = int(np.argmax(self.time > self.end_time))
#            self.time = self.time[:end_index]
#            self.U = self.U[:,:,:end_index]

        self.Nt = len(self.time)        

    def __read_experemetal_data(self):
        # Read wind tunnel measurements
        self.file_name = self.file_name[0]
        u = np.transpose(np.loadtxt(self.file_name +  '/U.txt'))
        self.Np = np.shape(u)[0]
        self.Nt = np.shape(u)[1]
        self.U = np.zeros((self.Np, self.no_cmpnt , self.Nt))
        self.U[:,0,:] = u
        self.U[:,1,:] = np.transpose(np.loadtxt(self.file_name +  '/V.txt'))
        self.U[:,2,:] = np.transpose(np.loadtxt(self.file_name +  '/W.txt'))
        self.locations = np.zeros((self.Np, self.no_cmpnt))
        self.z = np.loadtxt(self.file_name + '/location',comments='#')
        self.locations[:,2] = self.z
        self.f_max = 625.0
        self.dt = 1.0/(2.0*self.f_max)
        self.final_time = self.dt*self.Nt
        self.time = np.arange(0.0,self.final_time, self.dt)
        del u
#        self.end_time = 36.0 + self.start_time
        if(self.start_time != None):
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            self.U = self.U[:,:,start_index:]

        if(self.end_time != None):
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            self.U = self.U[:,:,:end_index]
        self.Nt = len(self.time)

    def __read_cfd_data(self):
        # Read cfd measurements        
        self.locations, self.time, u = connect_velocity_data_file_array(self.file_name)
        u = np.transpose(u)
        self.Np = np.shape(u)[1]
        self.Nt = np.shape(u)[2]
        self.U = np.zeros((self.Np, self.no_cmpnt , self.Nt))
        self.U[:,0,:] = u[0,:,:]
        self.U[:,1,:] = u[1,:,:]
        self.U[:,2,:] = u[2,:,:]
        self.z = self.locations[:,2]       
        self.dt = np.mean(np.diff(self.time))
        self.f_max = 1.0/(2.0*self.dt)
        self.final_time = self.time[-1]
        del u

        if(self.start_time != None):
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            self.U = self.U[:,:,start_index:]

        if(self.end_time != None):
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            self.U = self.U[:,:,:end_index]
        self.Nt = len(self.time)

    def __calculate_profiles(self):        
        self.u = np.zeros((self.Np, self.no_cmpnt, self.Nt))

        #Calculate the mean velocity profile.
        self.Uav = np.mean(self.U[:,0,:], axis=1)
        #Calculate the turbulence intensity.
        self.I = np.std(self.U, axis=2) # gets the standard deviation
        for i in range(self.no_cmpnt):
            self.I[:,i] = self.I[:,i]/self.Uav
        
        #Calculate the length scale profiles. 
        self.L = np.zeros((self.Np, self.no_cmpnt))
        for i in range(self.Np):
            for j in range(self.no_cmpnt):
                self.u[i,j,:] = self.U[i,j,:] - np.mean(self.U[i,j,:])
                self.L[i,j] = calculate_length_scale(self.u[i,j,:], self.Uav[i], self.dt)

#        Calculate the shear stress profiles. 
        self.uw_bar = np.zeros(self.Np)
        
        for i in range(self.Np):
            self.uw_bar[i] = np.cov(self.U[i,0,:], self.U[i,2,:])[0,1]

    def calculate_profiles(self):
        self.__read_file()
        self.__calculate_profiles()
            
    def get_Uav(self, z):
        from scipy import interpolate
        f = interpolate.interp1d(self.z, self.Uav)
        return f(z)
        
    def get_I(self, z):
        from scipy import interpolate
        f = interpolate.interp1d(self.z, self.I)
        return f(z)  
    
    def get_u_star(self, z_r):
        return self.get_Uav(z_r)/(2.5*np.log(z_r/(self.z_0/self.scale)))  
    
    def correct_mean(self, Uav):
        for i in range(self.Np):
            for j in range(self.no_cmpnt):
                self.U[i,j,:] = self.U[i,j,:] - np.mean(self.U[i,j,:]) + Uav[i,j]
        self.__calculate_profiles()
        
class Tap:
    name = ''
    index = -1
    x = 0
    y = 0
    z = 0
    cp = []
    face  = ''    # represents the side on with the probe is located
    neighbour_taps = []
    top_tap = ''    #Holds the tap at the top of this tap
    bottom_tap = ''      #Holds the tap thats at the bottom of this tap
    right_tap = ''       #Holds the tap thats at the right of this tap
    left_tap = ''        #Holds the tap thats at the left of this tap
    
    def __init__(self, index, name, x, y, z):
        self.index = index # index used to refer it in global tap array
        self.name = name #label used to name the tap in the presure measurment file
        self.x = x
        self.y = y
        self.z = z   
        
    def print_tap_info(self):
        print('Name = ' + self.name)
        print('Index = %d' % self.index)
        print('x = %f' % self.x)
        print('y = %f' % self.y)
        print('z = %f' % self.z)        
        print('Face = ' +  self.face)
        print('Neighbouring taps:')
        print('\t Top = '+ self.top_tap)
        print('\t Bottom = '+ self.bottom_tap)
        print('\t Right = '+ self.right_tap)
        print('\t Left = '+ self.left_tap)


    
class PIM:
    all_taps = []    
    
    top_taps = []
    north_taps = []
    east_taps = []
    south_taps = []
    west_taps = []
    cp_data = []
    broken_taps = []

    air_density = 1.25

       
    
    def __init__(self, setupfile):
        
        setup = open(setupfile, "r")
        lines = setup.readlines()
        
        for line in lines:
            atribute = line.strip()
            atribute = atribute.replace("\t", "")
            atribute = atribute.split(":")
            
            if atribute[0] == 'cp_file_name':
                self.Cp_file_path = atribute[1]
            if atribute[0] == 'tap_file_name':
                self.tap_file_name = atribute[1]
            if atribute[0] == 'wind_direction':
                self.wind_direction = float(atribute[1])
            if atribute[0] == 'building_height':
                self.building_height = float(atribute[1])
            if atribute[0] == 'building_width':
                self.building_width = float(atribute[1])
            if atribute[0] == 'building_depth':
                self.building_depth = float(atribute[1])
            if atribute[0] == 'z0':
                self.z0 = float(atribute[1])
            if atribute[0] == 'u_ref':
                self.u_ref = float(atribute[1])
            if atribute[0] == 'z_ref':
                self.z_ref = float(atribute[1])            
            if atribute[0] == 'gradient_height':
                self.gradient_height = float(atribute[1])
            if atribute[0] == 'gradient_wind_speed':
                self.gradient_wind_speed = float(atribute[1])
            if atribute[0] == 'scale':
                self.scale = float(atribute[1])
            if atribute[0] == 'broken_taps':
                self.broken_taps = atribute[1:]
            
        self.__create_taps()
        self.__read_cp_data()
#        self.__correct_cp_to_building_height()

    def __read_cp_data(self):
        cp_data = np.loadtxt(self.Cp_file_path)

        self.cp_data = np.zeros((self.tap_count, np.shape(cp_data)[1]))
        
        for i in range(self.tap_count):
            self.cp_data[i,:] = cp_data[i,:]
        
        #Take only the first 120second 
#        self.cp_data = self.cp_data[:,0:48000]

    def __create_taps(self):
        """
        Creates taps reading information from a text file. The tap coordinate should be formated as: 
        
        TapID       X-coord     Y-Coord     Z-coord
        -----       -------     -------     -------
        """
        tap_locations = open(self.tap_locations_path, "r")
        lines  = tap_locations.readlines()
        for line in lines:
            atribute = line.split('\t')
            theTap = tap(self.tap_count, atribute[0], float(atribute[1]), float(atribute[2]),  float(atribute[3]))
            self.taps.append(theTap)
            self.tap_count += 1   
     
        self.__assign_tap_faces()
        self.__create_neighbouring_taps()
        self.tap_count = len(self.taps)
        
    def __assign_tap_faces(self):    
        """
        Assignes the face where the tap is located as 'North', 'South', 'East' or 'West'.
        
        """
        north_x = 1.0e20
        south_x = -1.0e20
        east_y = -1.0e20
        west_y = 1.0e20
        top_z = -1.0e20

        #Indentify the faces
        for tapi in self.taps:
            if tapi.x < north_x:
                north_x = tapi.x
            if tapi.x > south_x:
                south_x = tapi.x
            if tapi.y > east_y:
                east_y = tapi.y
            if tapi.y < west_y:
                west_y = tapi.y
            if tapi.z > top_z:
                top_z = tapi.z
                  
        #Assign the face based on the coordinates.
        for tapi in self.taps:  
            if tapi.x == north_x:
                tapi.face = 'North'
                self.north.append(tapi.idx)
            if tapi.x == south_x:
                tapi.face = 'South'
                self.south.append(tapi.idx)
            if tapi.y == east_y:
                tapi.face = 'East'
                self.east.append(tapi.idx)
            if tapi.y  == west_y:
                tapi.face = 'West'
                self.west.append(tapi.idx)
            if tapi.z == top_z:
                tapi.face = 'Top'
                self.top.append(tapi.idx)            

                
    def __create_neighbouring_taps(self):    
        """
        Creates the neighbouring taps depending on the face where the tap is located.
                
        """
        dist = 1.0e20
        top = ''
        bottom = ''
        left = ''
        right = ''
        
        for tapi in self.taps:
            for tapj in self.taps:
                temp_dist =  calculate_distances(tapi.x, tapi.y, tapi.z, tapj.x, tapj.y, tapj.z)
                if temp_dist < dist:
                    if tapi.face == tapj.face:
                        if tapi.face == 'North':
                            if tapi.y == tapj.y and  tapi.z < tapj.z:
                                top = tapj.tag
                            if tapi.y == tapj.y and  tapi.z > tapj.z:
                                bottom = tapj.tag           
                            if tapi.z == tapj.z and  tapi.y > tapj.y:
                                right = tapj.tag          
                            if tapi.z == tapj.z and  tapi.y < tapj.y:
                                left = tapj.tag          
                        if tapi.face == 'South':
                            if tapi.y == tapj.y and  tapi.z < tapj.z:
                                top = tapj.tag
                            if tapi.y == tapj.y and  tapi.z > tapj.z:
                                bottom = tapj.tag           
                            if tapi.z == tapj.z and  tapi.y > tapj.y:
                                right = tapj.tag          
                            if tapi.z == tapj.z and  tapi.y < tapj.y:
                                left = tapj.tag  
                        if tapi.face == 'West':
                            if tapi.x == tapj.x and  tapi.z < tapj.z:
                                top = tapj.tag
                            if tapi.x == tapj.x and  tapi.z > tapj.z:
                                bottom = tapj.tag           
                            if tapi.z == tapj.z and  tapi.x > tapj.x:
                                right = tapj.tag          
                            if tapi.z == tapj.z and  tapi.x < tapj.x:
                                left = tapj.tag  
                        if tapi.face == 'East':
                            if tapi.x == tapj.x and  tapi.z < tapj.z:
                                top = tapj.tag
                            if tapi.x == tapj.x and  tapi.z > tapj.z:
                                bottom = tapj.tag           
                            if tapi.z == tapj.z and  tapi.x > tapj.x:
                                right = tapj.tag          
                            if tapi.z == tapj.z and  tapi.x < tapj.x:
                                left = tapj.tag  
                        if tapi.face == 'Top':
                            if tapi.x == tapj.x and  tapi.y < tapj.y:
                                top = tapj.tag
                            if tapi.x == tapj.x and  tapi.y > tapj.y:
                                bottom = tapj.tag           
                            if tapi.y == tapj.y and  tapi.x > tapj.x:
                                right = tapj.tag          
                            if tapi.y == tapj.y and  tapi.x < tapj.x:
                                left = tapj.tag  
            tapi.top = top
            tapi.bottom = bottom
            tapi.left = left
            tapi.right = right
            tapi.neighbours = [top, bottom, left, right]
            dist = 1.0e20
            top = ''
            bottom = ''
            left = ''
            right = ''            
            
            
    def __get_face_cp(self, face):
        
        if face == 'North':
            x_coord = np.zeros(len(self.north))
            y_coord = np.zeros(len(self.north))
            cp = np.zeros((len(self.north), np.shape(self.cp_data)[1]))
            for i in range(len(self.north)):
                x_coord[i] = self.taps[self.north[i]].y
                y_coord[i] = self.taps[self.north[i]].z
                cp[i,:] = self.cp_data[self.north[i],:]
        if face == 'West':
            x_coord = np.zeros(len(self.west))
            y_coord = np.zeros(len(self.west))
            cp = np.zeros((len(self.west), np.shape(self.cp_data)[1]))
            for i in range(len(self.west)):
                x_coord[i] = self.taps[self.west[i]].x
                y_coord[i] = self.taps[self.west[i]].z
                cp[i,:] = self.cp_data[self.west[i],:]
        if face == 'South':
            x_coord = np.zeros(len(self.south))
            y_coord = np.zeros(len(self.south))
            cp = np.zeros((len(self.south), np.shape(self.cp_data)[1]))
            for i in range(len(self.south)):
                x_coord[i] = self.taps[self.south[i]].y
                y_coord[i] = self.taps[self.south[i]].z
                cp[i,:] = self.cp_data[self.south[i],:]
        if face == 'East':
            x_coord = np.zeros(len(self.east))
            y_coord = np.zeros(len(self.east))
            cp = np.zeros((len(self.east), np.shape(self.cp_data)[1]))
            for i in range(len(self.east)):
                x_coord[i] = self.taps[self.east[i]].x
                y_coord[i] = self.taps[self.east[i]].z
                cp[i,:] = self.cp_data[self.east[i],:]                     
        
        return x_coord, y_coord, cp
    
    def find_tap_by_tag(self, tag):
        found_tap = -1 
        
        for tap in self.taps:
            if tap.tag == tag:
                found_tap = tap.idx
                break
        
        return found_tap
        
    def __correct_cp_to_building_height(self):
        
        #Correct the velocity using the velocity ratio of 
        #ESDU profile at two points, the gradient 
        uref = 20.0
        zref = 10.0

        esdu = ESDU(uref, zref, self.z0)
        z_gradient = 1.4732 # Gradient wind tunnle height
        u_grdient = esdu.get_V_z(self.scale*z_gradient)
        u_h = esdu.get_V_z(self.scale*self.height)
        
        corr = (u_grdient/u_h)**2.0
        
        print(corr)
        
        self.cp_data = self.cp_data*corr
        
    def __interpolate_tap_data(self, ngridx, ngridy, face):        
        from scipy.interpolate import griddata
        from scipy import interpolate
        x_coord, y_coord, cp = self.__get_face_cp(face)
        x_grid = np.linspace(np.min(x_coord), np.max(x_coord), ngridx)
        y_grid = np.linspace(np.min(y_coord), np.max(y_coord), ngridy)
        
        x_grid_new = np.linspace(-self.width/2.0, self.width/2.0, ngridx)
        y_grid_new = np.linspace(0.0, self.height, ngridy)
        z = np.mean(cp, axis=1)    
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = griddata((x_coord, y_coord), z, (X, Y), method='linear')       
        f = interpolate.interp2d(x_grid, y_grid, Z, kind='linear')
        X_new, Y_new = np.meshgrid(x_grid_new, y_grid_new)
        Z_new = f(x_grid_new, y_grid_new)
        return X_new, Y_new, Z_new
    
#    def __correct_broken_taps(self):
        
        #If some of the taps are broken, this fucntion corrects the 
        #pressure data by taking average between the nearest taps
        
#        for i in range(len(self.broken_taps)):
#            tap = self.__find_tap_by_tag(self.broken_taps[i])

        
#    def __find_neighbouring_taps(self, tap_tag):
        
#        tap = self.__find_tap_by_tag(tap_tag)
        
#        if tap.face == 'North': 
        
    def plot_faces_with_taps(self):
        import matplotlib.pyplot as plt    
        import matplotlib.patches as patches
        import matplotlib as mpl    
           
        fig = plt.figure(facecolor='white')
        markersize = 5
        linewidth = 1.25
        n_plots = 5
        
        ax = fig.add_subplot(1, n_plots, 1)

        ax.axis('off')
        border = patches.Rectangle((-self.width/2.0,0.0 ),self.width , self.height, linewidth=linewidth,edgecolor='k',facecolor='none')
        ax.add_patch(border)
        ax.set_title('North')
        ax.set_ylim(0.00, self.height)   
        for i in range(len(self.north)):
            ax.plot(self.taps[self.north[i]].y, self.taps[self.north[i]].z, 'k+', markersize=markersize)
#            ax.text(self.taps[self.north[i]].y, self.taps[self.north[i]].z, self.taps[self.north[i]].tag, fontsize=7,rotation=45)
            
        ax = fig.add_subplot(1, n_plots, 2)            
        ax.axis('off')
        border = patches.Rectangle((-self.depth/2.0,0.0 ), self.depth , self.height, linewidth=linewidth,edgecolor='k',facecolor='none')
        ax.add_patch(border)
        ax.set_title('West')
        ax.set_ylim(0.00,self.height) 
        for i in range(len(self.west)):
            ax.plot(self.taps[self.west[i]].x, self.taps[self.west[i]].z, 'k+', markersize=markersize)
#            ax.text(self.taps[self.west[i]].x, self.taps[self.west[i]].z, self.taps[self.west[i]].tag, fontsize=7,rotation=45)
#
#            
        ax = fig.add_subplot(1, n_plots, 3)            
        ax.axis('off')
        border = patches.Rectangle((-self.width/2.0,0.0 ), self.width , self.height, linewidth=linewidth,edgecolor='k',facecolor='none')
        ax.add_patch(border)
        ax.set_title('South')            
        ax.set_ylim(0.00,self.height)   
        for i in range(len(self.south)):
            ax.plot(self.taps[self.south[i]].y, self.taps[self.south[i]].z, 'k+', markersize=markersize)
#            ax.text(self.taps[self.south[i]].y, self.taps[self.south[i]].z, self.taps[self.south[i]].tag, fontsize=7,rotation=45)            
#
        ax = fig.add_subplot(1, n_plots, 4)           
        ax.axis('off')
        border = patches.Rectangle((-self.depth/2.0,0.0 ), self.depth , self.height, linewidth=linewidth,edgecolor='k',facecolor='none')
        ax.add_patch(border)
        for i in range(len(self.east)):
            ax.plot(self.taps[self.east[i]].x, self.taps[self.east[i]].z, 'k+', markersize=markersize)
#            ax.text(self.taps[self.east[i]].x, self.taps[self.east[i]].z, self.taps[self.east[i]].tag, fontsize=7,rotation=45)
#
#
#        ax = fig.add_subplot(1, n_plots, 5)            
#        ax.axis('off')
#        border = patches.Rectangle((-self.depth/2.0, -self.width/2.0), self.depth , self.width, linewidth=linewidth, edgecolor='r', facecolor='none')
#        ax.add_patch(border)        
#        ax.set_title('East')            
#        for i in range(len(self.top)):
#            ax.plot(self.taps[self.top[i]].x, self.taps[self.top[i]].y, 'k+', markersize=markersize)
#            ax.text(self.taps[self.top[i]].x, self.taps[self.top[i]].y, self.taps[self.top[i]].tag, fontsize=7,rotation=90)
        
        fig.set_size_inches(30/2.54, 75/2.54)    
        plt.tight_layout()
        plt.show()

 
    def plot_taps_and_walls(self):
        import matplotlib.pyplot as plt    
        import matplotlib.patches as patches
        import matplotlib as mpl    

#        tap1 = self.__find_tap_by_tag('609')
#        tap2 = self.__find_tap_by_tag('608')
#        
#        temp = self.cp_data[tap1.idx,:]
#        
#        self.cp_data[tap1.idx,:] = self.cp_data[tap2.idx,:]
#        self.cp_data[tap2.idx,:] = temp 

#        plt.plot(self.cp_data[tap1.idx,:])
#        plt.plot(self.cp_data[tap2.idx,:])
           
        fig = plt.figure(facecolor='white')
        markersize = 3
        linewidth = 0.25
        n_plots = 4
        ngridx = 70
        ngridy = 100
        ncontour = 20
        vmin = 0.0
        vmax = 1.0
        
        ax = fig.add_subplot(1, n_plots, 1)
        xx, yy, zz = self.__interpolate_tap_data(ngridx, ngridy, 'North')
        
#        ax.pcolorfast(xx, yy, zz, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        ax.contour(xx, yy, zz, ncontour, linewidths=0.5, colors='k', vmin=vmin, vmax=vmax)
        ctrplt = ax.contourf(xx, yy, zz, ncontour, cmap="RdBu_r", vmin=vmin, vmax=vmax)        
        plt.colorbar(ctrplt, ax=ax)


        for i in range(len(self.north)):
            ax.plot(self.taps[self.north[i]].y, self.taps[self.north[i]].z, 'k+', markersize=markersize)
            ax.axis('off')
            border = patches.Rectangle((-self.width/2.0,0.0 ),self.width , self.height, linewidth=linewidth,edgecolor='r',facecolor='none')
            ax.add_patch(border)
            ax.set_title('North')
            ax.set_ylim(0.00, self.height)            
#            ax.text(self.taps[self.north[i]].y, self.taps[self.north[i]].z, self.taps[self.north[i]].tag, fontsize=7,rotation=45)

            
        ax = fig.add_subplot(1, n_plots, 2)
        xx, yy, zz = self.__interpolate_tap_data(ngridx, ngridy, 'West')
        ax.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        ctrplt = ax.contourf(xx, yy, zz, 15, cmap="RdBu_r")
#        plt.colorbar(ctrplt, boundaries=np.linspace(0, 0.3, 15))
        for i in range(len(self.west)):
            ax.plot(self.taps[self.west[i]].x, self.taps[self.west[i]].z, 'k+', markersize=markersize)
            ax.axis('off')
            border = patches.Rectangle((-self.depth/2.0,0.0 ), self.depth , self.height, linewidth=linewidth,edgecolor='r',facecolor='none')
            ax.add_patch(border)
            ax.set_title('West')
            ax.set_ylim(0.00,0.75)            
#            ax.text(self.taps[self.west[i]].x, self.taps[self.west[i]].z, self.taps[self.west[i]].tag, fontsize=7,rotation=45)

            
        ax = fig.add_subplot(1, n_plots, 3)
        xx, yy, zz = self.__interpolate_tap_data(ngridx, ngridy, 'South')
        ax.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        ctrplt = ax.contourf(xx, yy, zz, 15, cmap="RdBu_r")
#        plt.colorbar(ctrplt, boundaries=np.linspace(0, 0.3, 15))
        for i in range(len(self.south)):
            ax.plot(self.taps[self.south[i]].y, self.taps[self.south[i]].z, 'k+', markersize=markersize)
            ax.axis('off')
            border = patches.Rectangle((-self.width/2.0,0.0 ), self.width , self.height, linewidth=linewidth,edgecolor='r',facecolor='none')
            ax.add_patch(border)
            ax.set_title('South')            
            ax.set_ylim(0.00,0.75)            
#            ax.text(self.taps[self.south[i]].y, self.taps[self.south[i]].z, self.taps[self.south[i]].tag, fontsize=7,rotation=45)            

        ax = fig.add_subplot(1, n_plots, 4)
        xx, yy, zz = self.__interpolate_tap_data(ngridx, ngridy, 'East')
        ax.contour(xx, yy, zz, 15, linewidths=0.5, colors='k')
        ctrplt = ax.contourf(xx, yy, zz, 15, cmap="RdBu_r")
#        plt.colorbar(ctrplt, boundaries=np.linspace(0, 0.3, 15))
        for i in range(len(self.east)):
            ax.plot(self.taps[self.east[i]].x, self.taps[self.east[i]].z, 'k+', markersize=markersize)
            ax.axis('off')
            border = patches.Rectangle((-self.depth/2.0,0.0 ), self.depth , self.height, linewidth=linewidth,edgecolor='r',facecolor='none')
            ax.add_patch(border)
            ax.set_title('East')            
            ax.set_ylim(0.00,0.75)            
#            ax.text(self.taps[self.east[i]].x, self.taps[self.east[i]].z, self.taps[self.east[i]].tag, fontsize=7,rotation=45)


#        ax = fig.add_subplot(1, n_plots, 5)
#        for i in range(len(self.top)):
#            ax.plot(self.taps[self.top[i]].x, self.taps[self.top[i]].y, 'k+', markersize=markersize)
#            ax.axis('off')
#            border = patches.Rectangle((-self.depth/2.0, -self.width/2.0), self.depth , self.width, linewidth=linewidth, edgecolor='r', facecolor='none')
#            ax.add_patch(border)
#            ax.text(self.taps[self.top[i]].x, self.taps[self.top[i]].y, self.taps[self.top[i]].tag, fontsize=7,rotation=90)
        
        fig.set_size_inches(30/2.54, 75/2.54)    
        plt.tight_layout()
        plt.show()
        
      
def readPSSfile(file_pssr,file_pssd):
    import scipy.io as sio
    
 
    #def readPSSfile(file_pssr,file_pssd):
    with open(file_pssr,'rb') as f:
        tester=np.multiply(np.fromfile(f,dtype='int16',count=-1),10/65536)
        
        
        
    file_pssd_loaded=sio.loadmat(file_pssd)['WTTDATALOG']
    channel_count=int(file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['StopAdd'][0,0][0][0]-file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['StartAdd'][0,0][0][0]+1)
    
    modules_used_count=int(file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['ModulesInUse'][0,0][0].size)
    modules_used=file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['ModulesInUse'][0,0][0]
    
    data=np.reshape(
        tester,
        (
            int(np.divide(np.divide(tester.size,modules_used_count),channel_count)),
            int(np.multiply(modules_used_count,channel_count))))
    
    
    
    for analog_module in file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['AnalogModules'][0,0][0]:
        analog_index=np.arange(analog_module-1,data.shape[1],modules_used_count)
        analog=data[:,analog_module-1:data.shape[1]:modules_used_count]
        data=np.delete(data,analog_index,axis=1)
        

        pth_modules=modules_used_count-file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['AnalogModules'][0,0][0].size
        
        if file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['ValidCal'][0,0][0][0]==1:
            
                    
            data=np.divide(data-np.tile(np.reshape(file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['CAL'][0,0][0][0]['Z'][0:pth_modules,:],((pth_modules)*channel_count),1).transpose(),(data.shape[0],1)),np.tile(np.reshape(file_pssd_loaded['APPSPE'][0,0][0,0][0][0][0]['MAN']['CAL'][0,0][0][0]['Q'][0:pth_modules,:],((pth_modules)*channel_count),1).transpose(),(data.shape[0],1)))
    
    
    cp_data=np.zeros((data.shape))
    
    
    for i in range(int(data.shape[1]/channel_count)-1):
        
        cp_data[:,i*channel_count:((i+1)*channel_count):1]=data[:,i:int(data.shape[1]):pth_modules]

    cp_data=np.array(cp_data,order='F')
    analog=np.array(analog)
    


    header=[]
    for mod in modules_used:
    
        for i in range(16):
            header.append(str(mod)+"{0:0>3}".format(i+1))

    return (cp_data,analog,header)
             
def fitt_loglaw_profile(velocities, heights):
    """
    Calculates the loglaw constants for a given descrte velocity data
    sample over heights.
    
    Args:
        velocities: the velocity profile.
        heights: the heighets where the velocities are extracted.
    Returns:
        u_star and z_0_r.
    
    Raises:
        KeyError: Raises an exception.
    """
    n_points  = len(velocities)
    u_fit = np.zeros(n_points)
    
    kappa = 0.4
    x = np.log(heights)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, velocities)
    u_star = slope*kappa
    z_0= np.exp(-intercept/slope)
    
    for i in range(n_points):
        u_fit[i] = (u_star/kappa)*np.log(heights[i]/z_0)
        
    return u_fit, z_0

def get_loglaw_profile(u_ref, z_ref, z_0, heights):
    """
    Calculates the loglaw for a given reference velocity, height and aerodynamicd 
    coefficient. 
    
    Args:
        velocities: the velocity profile.
        heights: the heighets where the velocities are extracted.
    Returns:
        Velocity profile.
    
    Raises:
        KeyError: Raises an exception.
    """
    n_points  = len(heights)
    velocities = np.zeros(n_points)
    
    kappa = 0.4
    u_star = kappa*u_ref/(np.log(z_ref/z_0))

    
    for i in range(n_points):
        velocities[i] = (u_star/kappa)*np.log(heights[i]/z_0)
        
    return velocities


def fitt_powerlaw_profile(velocities, heights, z_ref, u_ref):
    """
    Calculates the loglaw constants for a given descrte velocity data
    sample over heights.
    
    Args:
        velocities: the velocity profile.
        heights: the heighets where the velocities are extracted.
    Returns:
        u_star and z_0_r.
    
    Raises:
        KeyError: Raises an exception.
    """
    n_points  = len(velocities)
    u_fit = np.zeros(n_points)

    x = np.log(heights/z_ref)
    alpha, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(velocities/u_ref))

    for i in range(n_points):
        u_fit[i] = u_ref*np.power(heights[i]/z_ref, alpha)
        
    return u_fit, alpha

def von_karman_su(f, Uav, I, L):
    
    psd  = np.zeros(len(f))

    for i in range(len(f)):
        psd[i] = 4.0 * pow(I * Uav, 2.0)*(L / Uav) / pow(1.0 + 70.8*pow(f[i]*L/ Uav, 2.0), 5.0 / 6.0)

    return psd

def von_karman_sv(f, Uav, I, L):
    
    psd  = np.zeros(len(f))

    for i in range(len(f)):  
        psd[i] = 4.0 * pow(I*Uav, 2.0)*(L/Uav)*(1.0 + 188.4*pow(2.0 * f[i]*L/Uav, 2.0)) / pow(1.0 + 70.8*pow(2.0 * f[i]*L/Uav, 2.0), 11.0 / 6.0)

    return psd
    
def von_karman_sw(f, Uav, I, L):
    
    psd  = np.zeros(len(f))

    for i in range(len(f)):
        psd[i] = 4.0 * pow(I*Uav, 2.0)*(L/Uav)*(1.0 + 188.4*pow(2.0 * f[i]*L/Uav, 2.0)) / pow(1.0 + 70.8*pow(2.0 * f[i]*L/Uav, 2.0), 11.0 / 6.0)
    
    return psd
    
def von_karman_spectrum(f, Uav, I, L):
    
    psd  = np.zeros((3, len(f)))

    for i in range(len(f)):        
        psd[0, i] = 4.0 * pow(I[0]*Uav, 2.0)*(L[0]/Uav)/pow(1.0 + 70.8*pow(f[i]*L[0]/ Uav, 2.0), 5.0 / 6.0)
        psd[1, i] = 4.0 * pow(I[1]*Uav, 2.0)*(L[1]/Uav)*(1.0 + 188.4*pow(2.0 * f[i]*L[1]/Uav, 2.0)) / pow(1.0 + 70.8*pow(2.0 * f[i]*L[1]/Uav, 2.0), 11.0 / 6.0)
        psd[2, i] = 4.0 * pow(I[2]*Uav, 2.0)*(L[2]/Uav)*(1.0 + 188.4*pow(2.0 * f[i]*L[2]/Uav, 2.0)) / pow(1.0 + 70.8*pow(2.0 * f[i]*L[2]/Uav, 2.0), 11.0 / 6.0)

    return psd
    
def readVelocityProbes(fileName):
    """
    Created on Wed May 16 14:31:42 2018
    
    Reads velocity probe data from openfaom and return the probe location, time, and the velocity
    vector for each time step.
    
    @author: Abiy
    """
    probes = []
    U = []
    time  = []
    
    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(','')
                    line = line.replace(')','')
                    line = line.split()
                    probes.append([float(line[3]),float(line[4]),float(line[5])])
                else:
                    continue
            else: 
                line = line.replace('(','')
                line = line.replace(')','')
                line = line.split()
                try:
                    time.append(float(line[0]))
                except:
                    continue
                u_probe_i = np.zeros([len(probes),3])
                for i in  range(len(probes)):
                    u_probe_i[i,:] = [float(line[3*i + 1]),float(line[3*i + 2]),float(line[3*i + 3])]
                U.append(u_probe_i)
    
    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)
    
    return probes, time, U
    
def read_forces(fileName):
    """   
    Reads force data agregated over a surface from openfaom file and returns 
    origin, time, and the forces and moments vector for each time step.
    
    """
    origin = np.zeros(3)
    forces = []
    moments = []
    time = []
    
    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'): 
                if line.startswith('# CofR'): #Read the origin where the force are itegrated
                    line = line.replace('(','')
                    line = line.replace(')','')
                    line = line.split()
                    origin[0] = line[3] # x-coordinate 
                    origin[1] = line[4] # y-coordinate          
                    origin[2] = line[5] # z-coordinate
                else:
                    continue
            else: # Read only the pressure part of force and moments. Viscous and porous are ignored
                line = line.replace('(','')
                line = line.replace(')','')
                line = line.split()
                time.append(float(line[0]))
                forces.append([float(line[1]), float(line[2]), float(line[3])])
                moments.append([float(line[1 + 9]), float(line[2 + 9]), float(line[3 + 9])])
    
    time = np.asarray(time, dtype=np.float32)
    forces = np.asarray(forces, dtype=np.float32)
    moments = np.asarray(moments, dtype=np.float32)  
    
    return origin, time, forces, moments
    
def connect_forces_data(*args):
    """
    This functions takes names of different OpenFOAM forces measurments and connect
    them into one file removing overlaps if any. 
    Parameters
    ----------
    *args 
        List of file pathes of velocity data to be connected together. 
    Returns
    -------
    time, pressure
        Returns the veloicity time and velocity data of the connected file.
    """
    no_files  = len(args)
    connected_time = [] # Connected array of time 
    connected_F = []  # connected array of forces.
    connected_M = []  # connected array of moments.

    time1 = []
    F1    = []
    time2 = []
    F2    = []
    M1 = []
    M2 = []
    origin = []

    
    for i in range(no_files):         
        origin, time2, F2, M2 = read_forces(args[i])
        if i != 0:
            try:
                index = np.where(time2 == time1[-1])[0][0]
                index += 1
            except:
                # sys.exit('Fatal Error!: the pressure filese have time gap')
                index = 0 # Joint them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_F = np.concatenate((connected_F, F2[index:]))
            connected_M = np.concatenate((connected_M, M2[index:]))

        else:
            connected_time = time2
            connected_F = F2 
            connected_M = M2 

        time1 = time2
        F1 = F2
        M1 = M2

    return origin, connected_time, connected_F,connected_M
    

def readDisplacementProbes(fileName):
    """
    Created on Wed May 16 14:31:42 2018
    
    Reads velocity probe data from openfaom and return the probe location, time, and the velocity
    vector for each time step.
    
    @author: Abiy
    """
    probes = []
    disp = []
    time  = []
    
    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'):
                continue
            else: 
                line = line.replace('(','')
                line = line.replace(')','')
                line = line.split()
                time.append(float(line[0]))
                u_probe_i = np.zeros([len(probes),3])
                u_probe_i = [float(line[1]),float(line[2]),float(line[3])]
                disp.append(u_probe_i)
#                print(u_probe_i)
    
    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    disp = np.asarray(disp, dtype=np.float32)
    
    return probes, time, disp
    

def readPressureProbes(fileName):
    """
    Created on Wed May 16 14:31:42 2018
    
    Reads presure probe data from openfaom and return the probe location, time, and the pressure
    for each time step.
    
    @author: Abiy
    """
    probes = []
    p = []
    time  = []
    
    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(','')
                    line = line.replace(')','')
                    line = line.split()
                    probes.append([float(line[3]),float(line[4]),float(line[5])])
                else:
                    continue
            else: 
                line = line.split()
                time.append(float(line[0]))
                p_probe_i = np.zeros([len(probes)])
                for i in  range(len(probes)):
                    p_probe_i[i] = float(line[i + 1])
                p.append(p_probe_i)
    
    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    
    return probes, time, p

def calculate_cp(p, u_ref, p_ref=0, rho=1.0):
    return (p - p_ref)/(0.5*rho*u_ref**2.0)

def interpolate_time():
    """
    Creates interpolation a filed given. 
    Parameters
    ----------
    x: array
        x-coordiante of the points. 
        y-coordiante of the points. 
        z-coordiante of the points. 
    Returns
    -------
    distance
        Returns the distance of each point from the first point as an origin.
    """

def calculate_distances(x1, y1, z1, x2, y2, z2):
    """
    This function calculates distance between two points and return the result.
    Parameters
    ----------
    x1,y1,z1, x2,y2,z2: scalars
    Returns
    -------
    distance
        Returns the distance.
    """
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def calculate_path_distances(x, y, z=None):
    """
    This function calculates distance of each point on the path connecting points defined by
    x, y and z. The origin is set at the first point. 
    Parameters
    ----------
    x: array
        x-coordiante of the points. 
        y-coordiante of the points. 
        z-coordiante of the points. 
    Returns
    -------
    distance
        Returns the distance of each point from the first point as an origin.
    """
    n_points = len(x)
    distance = np.zeros(n_points)
    for i in range(n_points-1):
        if z is None:
            if abs(x[i]-x[i+1]) != 0 and abs(y[i]-y[i+1])!= 0: #Add cornner effects
                distance[i+1]= distance[i] + np.sqrt(2.0)*calculate_distances(x[i], y[i], 0, x[i+1], y[i+1], 0) 
            else:
                distance[i+1]= distance[i] + calculate_distances(x[i], y[i], 0, x[i+1], y[i+1], 0)
        else: 
            if abs(x[i]-x[i+1]) != 0 and abs(y[i]-y[i+1])!= 0: #Add cornner effects
                distance[i+1]= distance[i] + np.sqrt(2.0)*calculate_distances(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1])                
            else:
                distance[i+1]= distance[i] + calculate_distances(x[i], y[i], z[i], x[i+1], y[i+1], z[i+1]) 
   
    return distance

def connect_velocity_data(*args):
    """
    This functions takes names of different OpenFOAM velocity measurments and connect
    them into one file removing overlaps if any. All the probes must be in the same 
    location, otherwise an error might showup. 
    Parameters
    ----------
    *args 
        List of file pathes of velocity data to be connected together. 
    Returns
    -------
    time, pressure
        Returns the veloicity time and velocity data of the connected file.
    """
    no_files  = len(args)
    connected_time = [] # Connected array of time 
    connected_U = []  # connected array of pressure.

    time1 = []
    U1    = []
    time2 = []
    U2    = []
    probes = []
    
    for i in range(no_files):         
        probes, time2, U2 = readVelocityProbes(args[i])
        if i != 0:
            try:
                index = np.where(time2 == time1[-1])[0][0]
                index += 1
            except:
                # sys.exit('Fatal Error!: the pressure filese have time gap')
                index = 0 # Joint them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_U = np.concatenate((connected_U, U2[index:]))
        else:
            connected_time = time2
            connected_U = U2 

        time1 = time2
        U1 = U2
    return probes, connected_time, connected_U

def connect_velocity_data_file_array(args):
    """
    This functions takes names of different OpenFOAM velocity measurments and connect
    them into one file removing overlaps if any. All the probes must be in the same 
    location, otherwise an error might showup. 
    Parameters
    ----------
    *args 
        List of file pathes of velocity data to be connected together. 
    Returns
    -------
    time, pressure
        Returns the veloicity time and velocity data of the connected file.
    """
    no_files  = len(args)
    connected_time = [] # Connected array of time 
    connected_U = []  # connected array of pressure.

    time1 = []
    U1    = []
    time2 = []
    U2    = []
    probes = []
    
    for i in range(no_files):         
        probes, time2, U2 = readVelocityProbes(args[i])
        if i != 0:
            try:
                index = np.where(time2 == time1[-1])[0][0]
                index += 1
            except:
                # sys.exit('Fatal Error!: the pressure filese have time gap')
                index = 0 # Joint them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_U = np.concatenate((connected_U, U2[index:]))
        else:
            connected_time = time2
            connected_U = U2 

        time1 = time2
        U1 = U2
    return probes, connected_time, connected_U

def connect_pressure_data(*args):
    """
    This functions takes names of different OpenFOAM presure measurments and connect
    them into one file removing overlaps if any. All the probes must be in the same 
    location, otherwise an error might showup. 
    Parameters
    ----------
    *args 
        List of file pathes of pressure data to be connected together. 
    Returns
    -------
    time, pressure
        Returns the pressure time and pressure data of the connected file.
    """
    no_files  = len(args)
    connected_time = [] # Connected array of time 
    connected_p = []  # connected array of pressure.

    time1 = []
    p1    = []
    time2 = []
    p2    = []
    probes= []
               
    for i in range(no_files):            
        probes, time2, p2 = readPressureProbes(args[i])            
        if i != 0:
            try:
                index = np.where(time2 == time1[-1])[0][0]
                index += 1
            except:
                # sys.exit('Fatal Error!: the pressure filese have time gap')
                index = 0 # Joint them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_p = np.concatenate((connected_p, p2[index:]))
        else:
            connected_time = time2
            connected_p = p2 

        time1 = time2
        p1 = p2
    return probes, connected_time, connected_p

def write_abl_profile(z, uav, Iu, Iv, Iw, Lu, Lv, Lw, file_name):
       
    """
    Writes a given profile to a file delemiting with tab. 
        
    """   
    f = open(file_name,"w+")
    
    for i in range(len(z)):
        f.write('%f' % z[i])
        f.write("\t")
        f.write('%f' % uav[i])
        f.write("\t")
        f.write('%f' % Iu[i])
        f.write("\t")
        f.write('%f' % Iv[i])
        f.write("\t")
        f.write('%f' % Iw[i])
        f.write("\t")
        f.write('%f' % Lu[i])
        f.write("\t")
        f.write('%f' % Lv[i])
        f.write("\t")
        f.write('%f' % Lw[i])
        f.write('\n')
    
    f.close()

def write_open_foam_vector_field(p, file_name):
       
    """
    Writes a given profile to a file delemiting with tab. 
        
    """   
    f = open(file_name,"w+")
    f.write('%d' % len(p[:,2]))
    f.write('\n(')
    for i in range(len(p[:,2])):
        f.write('\n(%f %f %f)' % (p[i,0], p[i,1], p[i,2]))
    
    f.write('\n);')   
    f.close()


def calculate_length_scale_full(u, uav, dt):
    
     """
     Calculates the length scale of a velocity time history.
     This method computes the lenght using the full correlation array i.e tau=0 upto tau=t_max
    
     """   
     u = u - np.mean(u)

     corr = signal.correlate(u, u, mode='full')
    
     u_std = np.std(u)
     corr = corr[int(len(corr)/2):]/(u_std**2.0*len(u))        
     L  = uav*np.trapz(corr, dx=dt)
   
     return L

def calculate_length_scale(u, uav, dt):
    
     """
     Calculates the length scale of a velocity time history given.
    
     """   
     u = u - np.mean(u)

     corr = signal.correlate(u, u, mode='full')
    
     u_std = np.std(u)
    
     corr = corr[int(len(corr)/2):]/(u_std**2*len(u))
        
     loc = np.argmax(corr < 0)  

     corr = corr[:loc]
    
     L  = uav*np.trapz(corr, dx=dt)
   
     return L

def calculate_correlation_coef(u, dt):
    
     """
     Calculates the corelation coefficient.
    
     """   
     u = u - np.mean(u)

     corr = signal.correlate(u, u, mode='full')
    
     u_std = np.std(u)
    
     corr = corr[len(corr)/2:]/(u_std**2*len(u))
   
     return corr

def plot_mean_velocity(cfd_z, cfd_u,exp_z, exp_u, log_z, log_u, esdu_z, esdu_u, plt, legend1):
    """
    Plot the velocity profile in comparison with 
    ESDU log law, experemetal measurment and CFD.
    
    @author: abiy
    """
 # Plot mean velocity profile comparison

    # Define referce height and velocity
    # Approximatly equal with CAARC building height
    exp_ref_index = 8
    exp_z_ref = exp_z[exp_ref_index] 
    exp_u_ref = exp_u[exp_ref_index] 
#    cfd_u_ref = cfd_u[17] 
    cfd_u_ref = cfd_u[exp_ref_index] 
    # Normalize the height and mean velocity
    esdu_u= esdu_u/esdu_u[exp_ref_index]
    exp_u = exp_u/exp_u_ref
    exp_z = exp_z/exp_z_ref
    cfd_u = cfd_u/cfd_u_ref
    cfd_z = cfd_z/exp_z_ref
    log_u  = log_u/exp_u_ref
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 2
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0.5,1.25])
    ax.set_ylim([0,3.2])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(r'$Z/Z_{ref}$',font)
    ax.set_xlabel(r"$U_{av}/U_{ref}$",font)
    
    ax.plot(cfd_u, cfd_z,'ko', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(exp_u, exp_z,'ks', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(log_u, exp_z,'k-', markersize = markersize, linewidth=linewidth, markerfacecolor='None')
    ax.plot(esdu_u, exp_z,'r-', markersize = markersize, linewidth=linewidth,markerfacecolor='None')
#    ax.legend([legend1, 'Experiment', 'Loglaw($z_0 = 0.03$)', 'ESDU'], loc=0, fontsize=fontsize)
    ax.legend([legend1, 'Experiment', 'Loglaw', 'ESDU'], loc=0, fontsize=fontsize)

    plt.show()    
 
def plot_velocity_time_history( cfd_u, cfd_time, exp_u, exp_time, plt, label):
    """
    Plot the velocity profile in comparison with 
    ESDU log law, experemetal measurment and CFD.
    
    @author: abiy
    """
 # Plot mean velocity profile comparison

    # Define referce height and velocity
    # Approximatly equal with CAARC building height

    cfd_u = cfd_u - np.mean(cfd_u) 
    exp_u = exp_u - np.mean(exp_u)
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 1
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0,60])
    ax.set_ylim([-10,10])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(label,font)
    ax.set_xlabel(r"$Time(s)$",font)
    
    ax.plot(exp_time, exp_u,'b-', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)  
    ax.plot(cfd_time, cfd_u,'r-', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)

    ax.legend(['Experiment','Inflow' ], loc=0, fontsize=fontsize)
    plt.show()
 
def plot_turbulence_intensity( cfd_z, cfd_I,exp_z, exp_I, esdu_z, esdu_I, plt, label="u", legend1='LES-WT'):
    #Plot the turbulence intensity profile comparison

    exp_ref_index = 8
    exp_z_ref = exp_z[exp_ref_index] 
    cfd_z = cfd_z/exp_z_ref
    exp_z = exp_z/exp_z_ref


    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 2
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0,25])
    ax.set_ylim([0,3.2])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(r'$Z/Z_{ref}$',font)
    ax.set_xlabel(label,font)
    
    ax.plot(100*cfd_I, cfd_z,'ko', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(100*exp_I, exp_z,'ks', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(100*esdu_I, exp_z,'k', markersize = markersize, linewidth=linewidth,markerfacecolor='None', markeredgewidth=1.25)
    ax.legend([legend1, 'Experiment', 'ESDU'], loc=0, fontsize=fontsize)
    plt.show()
    
    
def plot_spectrum(cfd_u, cfd_Uav, cfd_I, cfd_L, cfd_dt, exp_u, exp_Uav, exp_I, exp_L,exp_dt, vonk_f, vonk_psd, label, plt):    
    
    from scipy import signal
    exp_f, exp_psd = signal.welch(exp_u, 1.0/exp_dt, nperseg=15000)
    cfd_f, cfd_psd = signal.welch(cfd_u, 1.0/cfd_dt, nperseg=10000)
        
    #Plot the turbulence intensity profile comparison
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 2
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0.01,1000])
    ax.set_ylim([1e-6,10])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(label,font)
    ax.set_xlabel(r'$f(Hz)$',font)
    
    ax.loglog(vonk_f,vonk_psd,'k-', markersize = markersize, linewidth=3, markerfacecolor='None', markeredgewidth=1.25)  
    ax.loglog(exp_f,exp_psd,'b-', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.loglog(cfd_f,cfd_psd,'r-', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)

    ax.legend(['vonKarman', 'Experiment','Inflow'], loc=3, fontsize=fontsize)
    plt.show()
    
def plot_mean_velocity_development(cfd_z, cfd_u, exp_z, exp_u, log_z, log_u, plt):

    ref_index = 8
    z_ref = log_z[ref_index] 
    u_ref = log_u[ref_index] 
    
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 2
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    plt.rcParams['font.family'] = "Times New Roman"
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0.6,1.2])
    ax.set_ylim([0,3.2])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(r'$Z/H$',font)
    ax.set_xlabel(r"$U_{av}/U_{H}$",font)
    
    ax.plot(cfd_u[0,:]/u_ref, cfd_z/z_ref,'k--',  linewidth=linewidth, markeredgewidth=1.25)
    ax.plot(cfd_u[1,:]/u_ref, cfd_z/z_ref,'k-.', linewidth=linewidth, markeredgewidth=1.25)
    ax.plot(cfd_u[2,:]/u_ref, cfd_z/z_ref,'k:', linewidth=linewidth, markeredgewidth=1.25)
    
    line1, = ax.plot(cfd_u[3,:]/u_ref, cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([2, 2, 10, 2])
    line1, = ax.plot(cfd_u[4,:]/u_ref, cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([1, 2, 10, 2])
    line1, = ax.plot(cfd_u[5,:]/u_ref, cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([2, 2, 2, 2, 10, 2])
    
    ax.plot(exp_u/u_ref, exp_z/z_ref,'ks', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(log_u/u_ref, exp_z/z_ref,'k-', markersize = markersize, linewidth=linewidth, markerfacecolor='None')
    ax.legend(['LES(x=0)', 'LES(x=2H)', 'LES(x=4H)', 'LES(x=6H)', 'LES(x=8H)', 'LES(x=10H)', 'Experiment', 'Loglaw'], loc=0, fontsize=fontsize)
    plt.show()
    
def plot_turbulence_intensity_development(cfd_z, cfd_I, exp_z, exp_I, esdu_z, esdu_I, plt, label):

    ref_index = 8
    z_ref = exp_z[ref_index] 
    
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1,1,1)
    fontsize = 20
    legendfontsize=15
    markersize = 9
    linewidth = 2
    
    font = {'family' : 'Times New Roman','weight' : 'normal', 'size'   : fontsize}
    ax.tick_params(direction='in', size=10)
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    #plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.rc('font', **font)
    plt.rc('axes', linewidth=1.5)
    ax.grid(linestyle='dotted', linewidth=1.25) 
    ax.set_xlim([0,30])
    ax.set_ylim([0,3.2])
    plt.rc('legend',fontsize=legendfontsize)
    ax.set_ylabel(r'$Z/H$',font)
    ax.set_xlabel(label,font)
    
    ax.plot(100*cfd_I[0,:], cfd_z/z_ref,'k--',  linewidth=linewidth, markeredgewidth=1.25)
    ax.plot(100*cfd_I[1,:], cfd_z/z_ref,'k-.', linewidth=linewidth, markeredgewidth=1.25)
    ax.plot(100*cfd_I[2,:], cfd_z/z_ref,'k:', linewidth=linewidth, markeredgewidth=1.25)
    
    line1, = ax.plot(100*cfd_I[3,:], cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([2, 2, 10, 2])
    line1, = ax.plot(100*cfd_I[4,:], cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([1, 2, 10, 2])
    line1, = ax.plot(100*cfd_I[5,:], cfd_z/z_ref,'k', linewidth=linewidth, markeredgewidth=1.25)
    line1.set_dashes([2, 2, 2, 2, 10, 2])
    
    ax.plot(100*exp_I, exp_z/z_ref,'ks', markersize = markersize, linewidth=linewidth, markerfacecolor='None', markeredgewidth=1.25)
    ax.plot(100*esdu_I, exp_z/z_ref,'k-', markersize = markersize, linewidth=linewidth, markerfacecolor='None')
    ax.legend(['LES(x=0)', 'LES(x=2H)', 'LES(x=4H)', 'LES(x=6H)', 'LES(x=8H)', 'LES(x=10H)', 'Experiment', 'ESDU'], loc=0, fontsize=fontsize)
    plt.show()

def calculate_coherency_function(f, C, x1, y1, z1, x2, y2, z2, Uav1, Uav2):
    dx = x2-x1
    dy = y2-y1
    dz = z2-z1
    
    Coh = np.zeros([3,])    
    Coh[0,] = np.exp(-2*f*np.sqrt(pow(dx*C[0,0],2) + pow(dy*C[0,1],2) + pow(dz*C[0,2],2))/(Uav1 + Uav2))
    Coh[1,] = np.exp(-2*f*np.sqrt(pow(dx*C[1,0],2) + pow(dy*C[1,1],2) + pow(dz*C[1,2],2))/(Uav1 + Uav2))
    Coh[2,] = np.exp(-2*f*np.sqrt(pow(dx*C[2,0],2) + pow(dy*C[2,1],2) + pow(dz*C[2,2],2))/(Uav1 + Uav2))

    return Coh

def coherency_function(f, C, x1, y1, z1, x2, y2, z2, Uav1, Uav2):
    dx = x2-x1
    dy = y2-y1
    dz = z2-z1
    n = len(f)
    Coh = np.zeros((n,3))  
    
    for i in range(n):
        Coh[i, 0] = np.exp(-2*f[i]*np.sqrt(pow(dx*C[0,0],2) + pow(dy*C[0,1],2) + pow(dz*C[0,2],2))/(Uav1 + Uav2))
        Coh[i, 1] = np.exp(-2*f[i]*np.sqrt(pow(dx*C[1,0],2) + pow(dy*C[1,1],2) + pow(dz*C[1,2],2))/(Uav1 + Uav2))
        Coh[i, 2] = np.exp(-2*f[i]*np.sqrt(pow(dx*C[2,0],2) + pow(dy*C[2,1],2) + pow(dz*C[2,2],2))/(Uav1 + Uav2))

    return Coh

def coherency_function_u(f, Cy, Cz, y1, z1, y2, z2, Uav1, Uav2):
    dy = y2-y1
    dz = z2-z1
    Uav = (Uav1 + Uav2)/2.0
    
    n = len(f)
    Coh = np.zeros(n)  
    for i in range(n):
        Coh[i] = np.exp(-f[i]*np.sqrt((dy*Cy)**2.0 + (dz*Cz)**2.0)/Uav)

    return Coh


#Calculates the spacial correlation given the the spectrum at two points and the coherency
def calculate_spacial_correlation(S1, S2, Coh):
    return np.dot(np.sqrt(np.multiply(S1, S2)), Coh)
   
def get_u_star(z_ref, u_ref, z0):
    return u_ref/(2.5*np.log(z_ref/z0))
    
def get_esdu_spectrum(f, Uav, I, L, z, h):
    A = 0.115*(1.0 + 0.315*(1.0-z/h)**6.0)**(2.0/3.0)
    a = 0.535 + 2.76*(0.138 - A)*0.68;  # alpha
    b1 = 2.357*a - 0.761   # beta1
    b2 = 1.0 - b1           # beta2
    nf = len(f)    
    S = np.zeros([3, nf])
    
    for i in range(nf):
        n = f[i]
        nu = n*L[0]/Uav
        F1 = 1.0 + 0.455*np.exp(-0.76*nu/a**-0.8)
        rSuu = b1*((2.987*nu/a)/(1 + (2*np.pi*nu/a)**2)**(5.0/6.0))+b2*((1.294*nu/a)/(1.0 +(np.pi*nu/a)**2.0)**(5.0/6.0))*F1
        varU = (I[0]*Uav)**2.0
        S[0,i] = rSuu*varU/n
    
        nv = n*L[1]/Uav;
        F2v = 1.0 + 2.88*np.exp(-0.218*nv/a**-0.9);
        rSvv = b1*(((2.987*(1+(8.0/3.0)*(4*np.pi*nv/a)**2.0)))*(nv/a)/(1.0 + (4*np.pi*nv/a)**2.0)**(11.0/6.0)) + b2*((1.294*nv/a)/((1.0 +(2*np.pi*nv/a)**2.0)**(5.0/6.0)))*F2v
        varV = (I[1]*Uav)**2.0;
        S[1,i] = rSvv*varV/n;
        
        nw = n*L[2]/Uav;
        F2w = 1.0 + 2.88*np.exp(-0.218*nw/(a**-0.9));
        rSww = b1*(((2.987*(1+(8.0/3.0)*(4*np.pi*nw/a)**2.0)))*(nw/a)/(1.0 + (4*np.pi*nw/a)**2.0)**(11.0/6.0)) + b2*((1.294*nw/a)/((1.0 +(2*np.pi*nw/a)**2.0)**(5.0/6.0)))*F2w
        varW = (I[2]*Uav)**2.0;
        S[2,i] = rSww*varW/n;
    
    return S    
    
def lieblin_blue(x):
    """
    Performs Lieblin Blue fitted peak values for a time series in 'x'.
    If the time series cannot be divided into 10 equal segments the remaining 
    part is discarded.
    """

    #Coefficient used for the Lieblein Blue fit
    a = [ 0.222867,  0.162308,  0.133845, 0.112868, 0.095636, 0.080618, 0.066988, 0.054193, 0.041748, 0.028929]
    b = [-0.347830, -0.091158, -0.019210, 0.022179, 0.048671, 0.066064, 0.077021, 0.082771, 0.083552, 0.077940]

    n_seg = 10 #Number of segments

    #Min and max of each segment
    x_max = np.zeros(n_seg)
    x_min = np.zeros(n_seg)

    #Number of time steps per each segment
    n_per_seg = len(x)/n_seg

    #Calculate the min and max of each segment.
    for i in range(n_seg):
        x_max[i] = np.amax(x[i*n_per_seg:(i+1)*n_per_seg])
        x_min[i] = np.amin(x[i*n_per_seg:(i+1)*n_per_seg])

    x_max = np.sort(x_max)      #sort in assending order
    x_min = -np.sort(-x_min)    #sort in decending order


    #Calculate the mode and dispertions
    u_max = np.dot(a,x_max)
    u_min = np.dot(a,x_min)
    d_max = np.dot(b,x_max)
    d_min = np.dot(b,x_min)


    #Calculate the peak based on Gambel distribution.
    x_peak_gb_max = u_max + d_max*np.log(n_seg)
    x_peak_gb_min = u_min + d_min*np.log(n_seg)

    #Calculate the stable peak using Lieblein Blue method.
    x_peak_lb_max = x_peak_gb_max + 0.5772*d_max
    x_peak_lb_min = x_peak_gb_min + 0.5772*d_min

    return x_peak_lb_max, x_peak_lb_min



class RME:  
    rho = 1.25

    def __init__(self, data_type, ref_height, ref_velocity, width, elevation, location, angle, cp_file_name):
        
        self.data_type = data_type # type of the simulatoin, experimetal or cfd
        self.ref_height = ref_height # Referencing height for the cp
        self.ref_velocity = ref_velocity
        self.width = width # width of the rme equipment(now assumed to be a cube) 
        self.elevation = elevation
        self.location = location
        self.angle = angle 
        self.scale = 50
        self.cp  = []
        self.p = []
        self.probes  = []
        self.n_faces  = 5
        self.n_taps  = 45
        self.dt = 0.0
        self.time  =  []
        self.n_times = 0
        self.start_time = None # start avaraging from time step ...
        self.end_time = None # start avaraging from time step ...
        self.cp_file_name = cp_file_name
        self.Cdx = []
        self.Cdy = []
        self.T0 = -1
        self.scale = 50.0 # Scale of the WT or CFD simulation. 

    def read_data(self): 

        if self.data_type == 'cfd':
            self.probes, self.time, self.p = connect_pressure_data(self.cp_file_name)
            self.dt = np.mean(np.diff(self.time)) 

            #Sampling time correction. 
            transient_region =   200       
            sampling_ratio = int(np.rint(0.0025/self.dt))
            self.p = self.p[transient_region:len(self.time):sampling_ratio,:]
            self.time = self.time[transient_region:len(self.time):sampling_ratio]
            self.fmax  = 1.0/self.dt
            self.calculate_cp()

        elif self.data_type == 'exp':
            self.cp = np.loadtxt(self.cp_file_name)
            self.fmax = 400.0
            self.dt = 1.0/self.fmax
            self.time = self.dt*np.arange(len(self.cp[:,0]))
        else: 
            print("Data type not recognized!")
        
        if(self.start_time != None):
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            self.cp = self.cp[start_index:,:]

        if(self.end_time != None):
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            self.cp = self.cp[:end_index,:]
        
        self.n_times = len(self.time)
        self.T0 = self.time[-1]

    def configure_tap_layout(self): 

        if self.elevation == 0:
            if self.data_type == 'cfd':
                self.N = [2, 1,  0,  5,  4,  3,  8,  7,  6]
                self.S = [9, 10, 11, 12, 13, 14, 15, 16, 17]
                self.E = [20,19, 18, 23, 22, 21, 26, 25, 24]
                self.W = [27,28, 29, 30, 31, 32, 33, 34, 35]
                self.T = [36,37, 38, 39, 40, 41, 42, 43, 44]

            elif self.data_type == 'exp':
                self.E = [6, 9,  12, 43, 45, 3,  35, 38, 41]-np.ones(9).astype(int)
                self.W = [44,2,  5,  37, 40, 42, 28, 31, 34]-np.ones(9).astype(int)
                self.N = [26,29, 32, 17, 20, 23, 8,  11, 14]-np.ones(9).astype(int)
                self.S = [33,36, 39, 24, 27, 30, 15, 18, 21]-np.ones(9).astype(int)
                self.T = [1, 10, 19, 4,  13, 22, 7,  16, 25]-np.ones(9).astype(int)
        
            self.taps = [self.N, self.E, self.S, self.W, self.T]
            self.n_taps  = 45            
            self.n_faces = 5
            self.n_taps_per_face = self.n_taps/self.n_faces
            self.tap_labels = range(1, self.n_taps + 1)
        else:
            if self.data_type == 'cfd':
                self.N = [2,  1,  0,  5,  4,  3,  8,  7,  6]
                self.S = [9,  10, 11, 12, 13, 14, 15, 16, 17]
                self.E = [20, 19, 18, 23, 22, 21, 26, 25, 24]
                self.W = [27, 28, 29, 30, 31, 32, 33, 34, 35]
                self.T = [36, 37, 38, 39, 40, 41, 42, 43, 44]
                self.B = [50, 51, 52, 49,     48, 47, 46, 45]

            elif self.data_type == 'exp':                
                self.N = [1, 2, 3, 4, 5, 6, 7, 8, 9]-np.ones(9).astype(int)                
                self.S = [19, 20, 21, 22, 23, 24, 25, 26, 27]-np.ones(9).astype(int)
                self.E = [10, 11, 12, 13, 14, 15, 16, 17, 18]-np.ones(9).astype(int)
                self.W = [28, 29, 30, 31, 32, 33, 34, 35, 36]-np.ones(9).astype(int)
                self.T = [37, 38, 39, 40, 41, 42, 43, 44, 45]-np.ones(9).astype(int)
                self.B = [46, 47, 48, 49,     50, 51, 52, 53]-np.ones(8).astype(int)

            self.taps = [self.N, self.E, self.S, self.W, self.T, self.B]
            self.n_taps  = 53            
            self.n_faces = 6
            self.n_taps_per_face = 9
            self.tap_labels = range(1, self.n_taps + 1)


    def calculate_cp(self): 
        if self.data_type == 'cfd': 
            #Reference dynamicd pressure
            p_dyn = 0.5*self.rho*self.ref_velocity**2.0   
            
            #Calculate referenced values
            self.cp = self.p/p_dyn

    def calculate_mean_rms_peak_cps(self):
        tap_count  = 0
        self.mean_cp = np.zeros(self.n_taps)
        self.rms_cp = np.zeros(self.n_taps)
        self.peak_pos_cp = np.zeros(self.n_taps)
        self.peak_neg_cp = np.zeros(self.n_taps)

        for j in range(self.n_faces):
            for i in range(len(self.taps[j])):
                self.mean_cp[tap_count] = np.mean(self.cp[:,self.taps[j][i]])
                self.rms_cp[tap_count] = np.std(self.cp[:,self.taps[j][i]])
                self.peak_pos_cp[tap_count], self.peak_neg_cp[tap_count] = lieblin_blue(self.cp[:,self.taps[j][i]])
                tap_count += 1
    
    def calculate_force_coefs(self):

        self.Cdx =  np.zeros(self.n_times)
        self.Cdy =  np.zeros(self.n_times)
        self.Cl =  np.zeros(self.n_times)

        for i in range(self.n_taps_per_face):
            self.Cdx += (self.cp[:,self.N[i]] - self.cp[:,self.S[i]])/float(self.n_taps_per_face )
            self.Cdy += (self.cp[:,self.E[i]] - self.cp[:,self.W[i]])/float(self.n_taps_per_face )
            self.Cl +=  (self.cp[:,self.T[i]])/float(self.n_taps_per_face)

        if self.elevation != 0:
            for i in range(len(self.B)):
                self.Cl -= (self.cp[:,self.B[i]])/float(len(self.B))

    def calculate_peak_forces(self):
        self.peak_pos_Cdx, self.peak_neg_Cdx = lieblin_blue(self.Cdx)
        self.peak_pos_Cdy, self.peak_neg_Cdy = lieblin_blue(self.Cdy)

    def calculate_area_averages(self):
        #Area averaged curves for each face
        avg_taps  = [1, 2, 3, 4, 6, 9]
        avg_taps_combs = [9,12, 6, 4, 4, 1]
        self.avg_taps_combs = avg_taps_combs
        n_avg_areas = len(avg_taps)        
        self.n_avg_areas = n_avg_areas
        self.avg_areas = np.zeros(n_avg_areas)
                   
        for i in range(n_avg_areas):
            self.avg_areas[i]  = 10.7639*(self.scale*self.scale)*avg_taps[i]*(self.width*self.width)/9.0 #Conversion factor times scale.scale

        self.peak_pos_avg_Cdx = np.zeros((n_avg_areas, 12))
        self.peak_neg_avg_Cdx = np.zeros((n_avg_areas, 12))
        self.peak_pos_avg_Cdy = np.zeros((n_avg_areas, 12))
        self.peak_neg_avg_Cdy = np.zeros((n_avg_areas, 12))
        self.peak_pos_avg_Cl = np.zeros((n_avg_areas, 12))
        self.peak_neg_avg_Cl = np.zeros((n_avg_areas, 12))

        avg_taps_list =  [[[0],[1],[2],[3],[4],[5],[6],[7],[8]],
                        [[0, 1],[0, 3],[1, 2],[1, 4],[2, 5],[3, 4],[3, 6],[4, 5],[4, 7],[5, 8],[6, 7],[7, 8]],
                        [[0, 1, 2],[3, 4, 5],[6, 7, 8],[0, 3, 6],[1, 4, 7],[2, 5, 8]],
                        [[0, 1, 3, 4],[1, 2, 4, 5],[6, 4, 6, 7],[4, 5, 7, 8]],
                        [[0, 1, 2, 3, 4, 5],[3, 4, 5, 6, 7, 8],[0, 1, 3, 4, 6, 7],[1, 2, 4, 5, 7, 8]],
                        [[0, 1, 2, 3, 4, 5, 6, 7, 8]]]

        avg_taps_mirror = [[[0],[1],[2],[3],[4],[5],[6],[7],[8]],
                        [[0, 1],[0, 3],[1, 2],[1, 4],[2, 5],[3, 4],[3, 6],[4, 5],[4, 7],[5, 8],[6, 7],[7, 8]],
                        [[0, 1, 2],[3, 4, 5],[6, 7, 8],[0, 3, 6],[1, 4, 7],[2, 5, 8]],
                        [[0, 1, 3, 4],[1, 2, 4, 5],[6, 4, 6, 7],[4, 5, 7, 8]],
                        [[0, 1, 2, 3, 4, 5],[3, 4, 5, 6, 7, 8],[0, 1, 3, 4, 6, 7],[1, 2, 4, 5, 7, 8]],
                        [[0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        
        #Create the mirror immage for area averaged forces
        for i in range(n_avg_areas):
            for j in range(avg_taps_combs[i]):
                for k in range(avg_taps[i]):
                    if avg_taps_mirror[i][j][k] == 0:
                        avg_taps_mirror[i][j][k] = 2
                    elif avg_taps_mirror[i][j][k] == 2:
                        avg_taps_mirror[i][j][k] = 0
                    if avg_taps_mirror[i][j][k] == 3:
                        avg_taps_mirror[i][j][k] = 5
                    elif avg_taps_mirror[i][j][k] == 5:
                        avg_taps_mirror[i][j][k] = 3
                    if avg_taps_mirror[i][j][k] == 6:
                        avg_taps_mirror[i][j][k] = 8
                    elif avg_taps_mirror[i][j][k] == 8:
                        avg_taps_mirror[i][j][k] = 6

        for i in range(n_avg_areas):
            for j in range(avg_taps_combs[i]):
                avg_Cdx_temp = np.zeros(self.n_times)
                avg_Cdy_temp = np.zeros(self.n_times)
                avg_Cl_temp = np.zeros(self.n_times)
                for k in range(avg_taps[i]):
                    avg_Cdx_temp += (self.cp[:,self.N[avg_taps_list[i][j][k]]] - self.cp[:,self.S[avg_taps_mirror[i][j][k]]])/float(avg_taps[i])
                    avg_Cdy_temp += (self.cp[:,self.E[avg_taps_list[i][j][k]]] - self.cp[:,self.W[avg_taps_mirror[i][j][k]]])/float(avg_taps[i])
                    if self.elevation == 0:
                        avg_Cl_temp += (self.cp[:,self.T[avg_taps_list[i][j][k]]])/float(avg_taps[i])
                    else:
                        cp_bot = []
                        if avg_taps_list[i][j][k] == 4:
                            cp_bot = (self.cp[:,self.B[3]] + self.cp[:,self.B[4]] + self.cp[:,self.B[1]] + self.cp[:,self.B[6]])/4.0
                        elif avg_taps_list[i][j][k] > 3 and avg_taps_list[i][j][k] != 4:
                            cp_bot = self.cp[:,self.B[avg_taps_list[i][j][k]-1]]
                        else:
                            cp_bot = self.cp[:,self.B[avg_taps_list[i][j][k]]]
                        avg_Cl_temp += (self.cp[:,self.T[avg_taps_list[i][j][k]]] - cp_bot)/float(avg_taps[i])

                self.peak_pos_avg_Cdx[i,j], self.peak_neg_avg_Cdx[i,j] = lieblin_blue(avg_Cdx_temp)
                self.peak_pos_avg_Cdy[i,j], self.peak_neg_avg_Cdy[i,j] = lieblin_blue(avg_Cdy_temp)
                self.peak_pos_avg_Cl[i,j], self.peak_neg_avg_Cl[i,j] = lieblin_blue(avg_Cl_temp)

        min_of_min_Cdx = np.zeros(n_avg_areas)
        max_of_max_Cdx = np.zeros(n_avg_areas)
        min_of_min_Cdy = np.zeros(n_avg_areas)
        max_of_max_Cdy = np.zeros(n_avg_areas)
        min_of_min_Cl = np.zeros(n_avg_areas)
        max_of_max_Cl = np.zeros(n_avg_areas)

        for i in range(n_avg_areas):
            max_of_max_Cdx[i] = np.max(self.peak_pos_avg_Cdx[i,0:avg_taps_combs[i]])
            min_of_min_Cdx[i] = np.min(self.peak_neg_avg_Cdx[i,0:avg_taps_combs[i]])
            max_of_max_Cdy[i] = np.max(self.peak_pos_avg_Cdy[i,0:avg_taps_combs[i]])
            min_of_min_Cdy[i] = np.min(self.peak_neg_avg_Cdy[i,0:avg_taps_combs[i]])
            max_of_max_Cl[i] = np.max(self.peak_pos_avg_Cl[i,0:avg_taps_combs[i]])
            min_of_min_Cl[i] = np.min(self.peak_neg_avg_Cl[i,0:avg_taps_combs[i]])

        max_p_Cdx =  np.poly1d(np.polyfit(np.log10(self.avg_areas),max_of_max_Cdx, 1))
        min_p_Cdx =  np.poly1d(np.polyfit(np.log10(self.avg_areas),min_of_min_Cdx, 1))

        max_p_Cdy =  np.poly1d(np.polyfit(np.log10(self.avg_areas),max_of_max_Cdy, 1))
        min_p_Cdy =  np.poly1d(np.polyfit(np.log10(self.avg_areas),min_of_min_Cdy, 1))

        max_p_Cl =  np.poly1d(np.polyfit(np.log10(self.avg_areas),max_of_max_Cl, 1))
        min_p_Cl =  np.poly1d(np.polyfit(np.log10(self.avg_areas),min_of_min_Cl, 1))


        max_of_max_Cdx = max_p_Cdx(np.log10(self.avg_areas))
        min_of_min_Cdx = min_p_Cdx(np.log10(self.avg_areas))

        max_of_max_Cdy = max_p_Cdy(np.log10(self.avg_areas))
        min_of_min_Cdy = min_p_Cdy(np.log10(self.avg_areas))

        max_of_max_Cl = max_p_Cl(np.log10(self.avg_areas))
        min_of_min_Cl = min_p_Cl(np.log10(self.avg_areas))        

        self.max_peak_pos_avg_Cdx = max_of_max_Cdx
        self.min_peak_neg_avg_Cdx = min_of_min_Cdx
        self.max_peak_pos_avg_Cdy = max_of_max_Cdy
        self.min_peak_neg_avg_Cdy = min_of_min_Cdy
        self.max_peak_pos_avg_Cl = max_of_max_Cl
        self.min_peak_neg_avg_Cl = min_of_min_Cl

        self.max_peak_abs_avg_Cdx = np.maximum(abs(self.max_peak_pos_avg_Cdx), abs(self.min_peak_neg_avg_Cdx))
        self.max_peak_abs_avg_Cdy = np.maximum(abs(self.max_peak_pos_avg_Cdy), abs(self.min_peak_neg_avg_Cdy))
        self.max_peak_abs_avg_Cl =  np.maximum(abs(self.max_peak_pos_avg_Cl), abs(self.min_peak_neg_avg_Cl))

    def calculate_all(self):
        self.read_data()
        self.configure_tap_layout()
        self.calculate_mean_rms_peak_cps()
        self.calculate_force_coefs()
        self.calculate_peak_forces()
