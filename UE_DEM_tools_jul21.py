import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import rasterio
import pickle
import os
import subprocess
import geopandas
import pandas
import shapely
from tqdm import tqdm
from osgeo import gdal
import cv2

def rgb2gray(rgb):
    if len(rgb.shape) < 3:
        return rgb
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def map2grid(pos, cell_size):
    return (int(pos[0] / cell_size), int(pos[1] / cell_size))

class DEM():
    def __init__(self, heightmap_path, vmin=0, vmax=1, style="gascola", top_right_augment = 0, scaler=4, pixel2metre=0.5):
        self.style = style
        self.heightmap_path = heightmap_path # contains the terrain heightmap
        # it is possible that it expects that 1 px = 1 metre
        #self.heightmap = np.flipud(img.imread(heightmap_path))
        self.heightmap = Image.open(heightmap_path).convert("L")
        shape = np.shape(self.heightmap)
        self.heightmap = self.heightmap.resize((shape[1]//scaler, shape[0]//scaler))
        self.heightmap -= np.min(self.heightmap)
        self.heightmap = cv2.copyMakeBorder(self.heightmap, top_right_augment, 0, 0, top_right_augment, cv2.BORDER_REPLICATE)
        self.top_right_augment = top_right_augment
        # these seem to set the height manually 0 and 1 leave unchanged
        self.vmin = vmin # vmin and vmax represent metres
        self.vmax = vmax

        self.bounds = None
        self.pixel2metre = pixel2metre*scaler
        self.scale_vertical(self.vmin, self.vmax)
        plt.imshow(self.heightmap)
        plt.savefig("heightmap_grayscale_preview.png")

        self.viewshed = None


    def scale_vertical(self, vmin, vmax):
        """
        :param vmin: min height enforced
        :param vmax: max height enforced
        :return: Nothing, just modifies heightmap attribute
        """
        zmin = np.min(self.heightmap)
        zmax = np.max(self.heightmap)
        self.heightmap = (self.heightmap-zmin)/zmax
        self.heightmap = self.heightmap*(vmax-vmin)+vmin

    def terrain2map(pos, cell_size=0.59, style="horizontal"):
        if style == "horizontal" or style == "gascola":
            return ((pos[0] - 173) * cell_size, (pos[1] - 27) * cell_size)
        elif style == "vertical":
            return ((pos[1] - 173) * cell_size, (1920 - pos[0] - 27) * cell_size)
        elif style == "ntc":
            return ((pos[0] - 335) * cell_size, (pos[1] - 41) * cell_size)

    def map2terrain(pos, cell_size=0.59, style="horizontal"):
        if style == "horizontal" or style == "gascola":
            return (int(round(173 + pos[0] / cell_size)), int(round(27 + pos[1] / cell_size)))
        elif style == "vertical":
            return (int(round(173 + pos[1] / cell_size)), int(round(1920 - 27 - pos[0] / cell_size)))
        elif style == "ntc":
            return (int(round(335 + pos[0] / cell_size)), int(round(41 + pos[1] / cell_size)))

    def geocode(self, bounds=None, projstr=None):

        if bounds is None:
            self.bounds = [-width/2, -height/2, width/2, height/2]
        else:
            self.bounds = bounds

        if projstr == None:
            projstr = "+proj=utm +zone=8 +ellps=WGS84 +units=m +no_defs"
        # read rasterio documentation to understand these settings
        self.height, self.width = np.shape(self.heightmap)
        print("width: ", self.width, "height: ", self.height)
        bounds = [bounds[0], bounds[2], bounds[1] + self.top_right_augment*self.pixel2metre, bounds[3] + self.top_right_augment*self.pixel2metre]
        print(*bounds, "west south east north")
        # affine = rasterio.transform.from_bounds(bounds[2], bounds[1], bounds[3], bounds[0], width, height)
        affine = rasterio.transform.from_bounds(*bounds, width=self.width,
                                                height=self.height)

        print(affine)
        if self.style == "gascola":
            print(rasterio.transform.rowcol(affine, 535, 1116), "should be (approx): ", (0, 1080))
            print(rasterio.transform.rowcol(affine, 0, 1116), "should be (approx): ", (0, 0))
            print(rasterio.transform.rowcol(affine, 535, 0), "should be (approx): ", (1920, 1080))
            print(rasterio.transform.rowcol(affine, 0, 0), "should be (approx): ", (1920, 0))
        elif self.style == "ntc":
            print(rasterio.transform.rowcol(affine, 535, 1116), "should be (approx): ", (0, 1080))
            print(rasterio.transform.rowcol(affine, 0, 1116), "should be (approx): ", (0, 0))
            print(rasterio.transform.rowcol(affine, 535, 0), "should be (approx): ", (1920, 1080))
            print(rasterio.transform.rowcol(affine, 0, 0), "should be (approx): ", (1920, 0))

        #memfile = rasterio.MemoryFile()
        #rst = memfile.open(driver='GTiff', width=width, height=height,
        #                   count=1, dtype=rasterio.float32,
        #                   transform=affine)
        #rst.write(self.heightmap.astype(rasterio.float32), 1)
        #rst.close

        self.profile = {'driver':'GTiff', 'width':self.width, 'height':self.height,
                           'count':1, 'dtype':rasterio.float32,
                           'crs':projstr,
                           'transform':affine}
        # print((535, 1116)*self.profile['transform'])
        self.tif_path = self.heightmap_path.split(".")[0]+".tif"
        with rasterio.open(self.tif_path, 'w', **self.profile) as dst:
            print("Writing heightmap to "+self.tif_path)
            dst.write(self.heightmap.astype(rasterio.float32), 1)

    def compute_viewshed(self, x, y, target_height=None, observer_height=None, maximum_distance=200):
        # This computes the which locations in the DEM are viewable from the x and
        # y location. Optionally, a target height and observer height can be given as well.
        # https://gdal.org/programs/gdal_viewshed.html
        cmd = 'gdal_viewshed -ox ' + str(x) + ' -oy '+ str(y)# + ' -md ' + str(maximum_distance)
        if target_height is not None:
            cmd += " -tz "+str(target_height)
        if observer_height is not None:
            cmd += " -oz "+str(observer_height)

        if self.tif_path is None:
            print("Run geocode first!!")
            return
        else:
            cmd += " "+self.tif_path + " viewshed_temp.tif"

        # print(cmd)
        subprocess.call([cmd], shell=True)

        # Now load the data from the tif.
        with rasterio.open("viewshed_temp.tif") as ds:
            viewshed = ds.read(1)/255.

        # viewshed = gdal.ViewshedGenerate()
        #clean up and return the viewshed matrix. It has the same transform
        # as the dem object used, so a matrix is enough
        if self.viewshed is None:
            self.viewshed = viewshed
        else:
            self.viewshed += viewshed
        # print(self.viewshed.shape, np.max(self.viewshed), np.min(self.viewshed))
        # os.remove('viewshed_temp.tif')
        # plt.imshow(viewshed)
        # plt.savefig(str(x)+str(y)+".png")
        # plt.close()
        path = "./to_gif/" + self.style + "_"+ "/"
        if not os.path.exists(path):
            os.makedirs(path)
        Image.fromarray((viewshed * 255).astype(np.uint8)).save(path + str(x)+"_"+str(y)+".png")
        return viewshed

    def height_at_point(self, UE_rx, UE_ry):
        # This returns the height at a point in UE coordinates. The point of
        # interest is sampled to the nearest pixel coordinate, so it will have a
        # horizontal precision of 1 meter.
        i, j = rasterio.transform.rowcol(self.profile['transform'], UE_rx, UE_ry)
        print(UE_rx, UE_ry)
        print(i, j)
        z = self.heightmap[i,j]
        return z

class RoboGrid():
    def __init__(self, dem, xinc=30, yinc=30, xn=16, yn=16):
        # DEM is an object of the DEM class. It has the matrix of the
        # heightmap and the path to the heightmaps geotif file.

        self.dem = dem
        self.xinc = xinc
        self.yinc = yinc
        self.xn = xn
        self.yn = yn
        self.robogrid = {}

        self.__compute_robogrid_nodes__()

    def __compute_robogrid_nodes__(self):
        # This starts out by defining the robogrid nodes in UE space
        # and then gets the IM_coordinates afterwards. This is to ensure that
        # the robogrid nodes are spaced in even UE increments.

        # node = 0
        ht_coord = 0
        wt_coord = 0
        # for y_idx in np.arange(self.dem.bounds[2]+5, self.dem.bounds[3]-self.yinc, self.yinc):
        # print(np.arange(0, self.dem.bounds[3]- self.yinc, self.yinc))
        # print(self.yinc//2, self.dem.bounds[3], self.yinc)
        for y_idx in np.arange(self.yinc//2, self.dem.bounds[3] + self.dem.pixel2metre*self.dem.top_right_augment, self.yinc):#, self.yinc):
            # going from -15.93 to 1116.87 in 37 steps
            wt_coord=0
            # for x_idx in np.arange(self.dem.bounds[0]+5, self.dem.bounds[1]-self.xinc, self.xinc):
            for x_idx in np.arange(self.xinc//2, self.dem.bounds[1] + self.dem.pixel2metre*self.dem.top_right_augment, self.xinc):#, self.xinc):
                # GOING FROM  -102.07 to 535.13
                UE_rx = x_idx #(x_idx*self.xinc+self.xinc/2.) #- (width-self.xinc*self.xn)
                UE_ry = y_idx #(y_idx*self.yinc+self.yinc/2.) #- (height-self.yinc*self.yn)/2.

                # rx and ry are now in image coordinatesr. Need to put them in UE coordinates
                IM_rx, IM_ry = rasterio.transform.rowcol(self.dem.profile['transform'], UE_rx, UE_ry)
                print("WORLD: ", UE_rx, UE_ry, "IMAGE: ", IM_rx, IM_ry)
                # IM_rx, IM_ry = rasterio.transform.i, j = (self.dem.profile['transform'], UE_rx, UE_ry)
                z = self.dem.heightmap[IM_rx, IM_ry]
                # node = (wt_coord, ht_coord)
                node = map2grid((UE_rx, UE_ry), self.yinc)
                # print("OG: ", node)
                # print("Function: ", map2grid((UE_rx, UE_ry), self.yinc))
                self.robogrid[node] = {'IM_coords':(IM_rx, IM_ry), 'UE_coords':(UE_rx, UE_ry), 'z':z}
                wt_coord+=1
            ht_coord+=1

        self.xn = wt_coord
        self.yn = ht_coord
        print("xn, wt: ", self.xn, "yn, ht: ", self.yn)

    def compute_node_viewshed(self, observer_height=None, target_height=None, nodes='all'):
        # Nodes can either be all or a list of the individual nodes to compute the
        # viewshed for.
        if nodes == 'all':
            nodes = list(self.robogrid.keys())
        for node in tqdm(nodes):
            rx, ry = self.robogrid[node]['UE_coords']
            viewshed = self.dem.compute_viewshed(rx, ry, observer_height, target_height)

            # I need to make a raster in memory of the viewshed matrix.
            memfile = rasterio.MemoryFile()
            height, width = np.shape(self.dem.heightmap)
            affine = self.dem.profile['transform']
            rst = memfile.open(driver='GTiff', width=width, height=height,
                               count=1, dtype=rasterio.float32,
                               transform=affine)
            rst.write(viewshed.astype(rasterio.float32), 1)

            # Now find the neighboring nodes in each direction and compute the
            # average visibility for each. This is going to get wiggy!!!
            self.robogrid[node]['viewshed'] = {0:{}, 1:{}, 2:{}, 3:{}}
            for direction in range(4):
                dir_nodes = self.get_directional_nodes(node, direction)

                for dir_node in dir_nodes:
                    # Get the bounds of each node.
                    edges = self.__get_node_edges__(dir_node)
                    x1, x2 = np.min(edges[:,0]), np.max(edges[:,0])
                    y1, y2 = np.min(edges[:,1]), np.max(edges[:,1])
                    window_bounds = [x1,y1,x2,y2]
                    # print("Window bouns: ", window_bounds)
                    # Now get the average of the visibility inside the node bounds
                    window =  rst.window(*window_bounds)
                    view_node = rst.read(1, window=window)
                    self.robogrid[node]['viewshed'][direction][dir_node] = np.average(view_node)
                    print(dir_node, np.average(view_node))

            rst.close()
            if self.dem.viewshed is not None:
                del viewshed

    def __get_node_edges__(self, node):
        # print("node: ", node)
        node = (node % self.xn, int(node/self.xn))
        # print("node_tuple: ", node)
        rx, ry = self.robogrid[node]["UE_coords"]
        edges = []
        edges.append([rx-self.xinc/2, ry+self.yinc/2])
        edges.append([rx+self.xinc/2, ry+self.yinc/2])
        edges.append([rx+self.xinc/2, ry-self.yinc/2])
        edges.append([rx-self.xinc/2, ry-self.yinc/2])
        edges.append([rx-self.xinc/2, ry+self.yinc/2])
        return np.array(edges)

    def plot_shapefiles(self, ax=None, cmap='jet', legend=True):
        if ax is None:
            fig, ax = plt.subplots()

        multi_poly = []
        key_val = []
        for node in self.robogrid.keys():
            z = self.robogrid[node]['z']
            #edges = np.array(h3.h3_to_geo_boundary(hex_address)) # Returns lat lon!!
            edges = self.__get_node_edges__(node)
            poly = []
            for i in range(np.shape(edges)[0]):
                poly.append([edges[i,0], edges[i,1]])

            multi_poly.append(shapely.geometry.Polygon(poly))
            key_val.append({'MD':z})

        node_shp = geopandas.GeoDataFrame(pandas.DataFrame(key_val), geometry = multi_poly, crs="+init=epsg:4326")
        node_shp.plot(ax=ax, edgecolor='white', facecolor='none')

    # def get_directional_nodes(self,node,d):
    def get_directional_nodes(self, node, d):
        if isinstance(node, int):
            l = node % self.xn
            h = (node - l) / self.yn
            # print("Calced: l,h", l,h)
        else:
            l, h = node
            # print("Received: l,h", l,h)

        if(d == 1):
            # North
            # print("1, North")
            ls = np.array([l-1,l,l+1,l-2,l-1,l,l+1, l+2, l-3,l-2,l-1,l,l+1,l+2, l+3])
            hs = np.array([h+1,h+1,h+1,h+2,h+2,h+2,h+2, h+2, h+3,h+3,h+3,h+3,h+3,h+3, h+3])

            # ls = np.array([l-1,l,l-2,l-1,l,l+1,l-3,l-2,l-1,l,l+1,l+2])
            # hs = np.array([h+1,h+1,h+2,h+2,h+2,h+2,h+3,h+3,h+3,h+3,h+3,h+3])
           # non_zero_idx_t = np.array([i+n1-1,i+n1,i+2*n1,i+2*n1-1,i+2*n1-2,
#                                     i+2*n1+1,i+3*n1-3,i+3*n1-2,i+3*n1-1,
#                                     i+3*n1,i+3*n1+1,i+3*n1+2])
        elif(d == 0):
            # EAST
            # print("0, EAST")
            ls = np.array([l+1,l+1, l+1, l+2,l+2,l+2,l+2,l+2, l+3, l+3,l+3,l+3,l+3,l+3,l+3])
            hs = np.array([h+1,h, h-1,   h+2,h+1,h,h-1,h-2,   h+3,h+2,h+1,h,h-1,h-2, h-3])

            # ls = np.array([l+1,l+1,l+2,l+2,l+2,l+2,l+3,l+3,l+3,l+3,l+3,l+3])
            # hs = np.array([h+1,h,h+2,h+1,h,h-1,h+3,h+2,h+1,h,h-1,h-2])
#            non_zero_idx_t = np.array([i+1,i+n1+1,i+2,i+2-n1,i+2+n1,i+2+2*n1,
#                                     i+3,i+3-n1,i+3-2*n1,i+3+n1,i+3+2*n1,i+3+3*n1])
        elif(d == 3):
            # SOUTH
            # print("3, South")
            ls = np.array([l-1,l,l+1,  l-2,l-1,l,l+1,l+2,  l-3,l-2,l-1,l,l+1,l+2,l+3])
            hs = np.array([h-1,h-1,h-1,h-2,h-2,h-2,h-2,h-2,h-3,h-3,h-3,h-3,h-3,h-3,h-3])

            # ls = np.array([l,l+1,l-1,l,l+1,l+2,l-2,l-1,l,l+1,l+2,l+3])
            # hs = np.array([h-1,h-1,h-2,h-2,h-2,h-2,h-3,h-3,h-3,h-3,h-3,h-3])
#            non_zero_idx_t = np.array([i-n1,i-n1+1,i-2*n1,i-2*n1-1,i-2*n1+1,i-2*n1+2,
#                                     i-3*n1,i-3*n1-1,i-3*n1-2,i-3*n1+1,i-3*n1+2,i-3*n1+3])
        elif(d == 2):
            # WEST
            # print("2, West")
            ls = np.array([l-1,l-1,l-1,l-2,l-2,l-2,l-2,l-2,l-3,l-3,l-3,l-3,l-3,l-3,l-3])
            hs = np.array([h-1,h,h+1,h-2,h-1,h,h+1,h+2,h-3,h-2,h-1,h,h+1,h+2,h+3])

            # ls = np.array([l-1,l-1,l-2,l-2,l-2,l-2,l-3,l-3,l-3,l-3,l-3,l-3])
            # hs = np.array([h-1,h,h-2,h-1,h,h+1,h-3,h-2,h-1,h,h+1,h+2])
#            non_zero_idx_t = np.array([i-1,i-1-n1,i-2,i-2-n1,i-2-2*n1,i-2+n1,
#                                     i-3,i-3-n1,i-3-2*n1,i-3-3*n1,i-3+n1,i-3+2*n1])
        else:
            print('wrong d parameter')

        non_zero_idx = []
        noise_var = []
        count = 0
        for ii in range(0,len(ls)):
            if(ls[ii]<self.xn and ls[ii]>=0 and hs[ii]<self.yn and hs[ii]>=0):
                pos = int(hs[ii]*self.xn+ls[ii])
                non_zero_idx.append(pos)
                #noise_var.append(self.noise_vec[ii])
                count = count+1
        # x = np.zeros((count,self.xn*self.yn))
        # for jj in range(count):
        #     x[jj,non_zero_idx[jj]] = 1
#        print(i,d)
#        print(np.array(non_zero_idx))

        return np.array(non_zero_idx)

    def interpolate_trajectory(self, trajectory, dx):
        # This assumes that gmt is installed
        # This takes a dictionary with keys of x, y points of interest
        # It interpolates the line to have points spaced out a distance of dx
        # apart. Then it uses grdtrack to sample the z values of the DEM at these points
        #
        x_interp = np.array([])
        y_interp = np.array([])
        dir = np.array([])
        for idx in range(len(trajectory['x'])-1):
            x1, x2 = trajectory['x'][idx], trajectory['x'][idx+1]
            y1, y2 = trajectory['y'][idx], trajectory['y'][idx+1]

            line_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
            N = int(line_length/dx)
            xs = np.linspace(x1, x2, N)
            ys = np.linspace(y1, y2, N)
            x_interp = np.append(x_interp, xs)
            y_interp = np.append(y_interp, ys)

            for i in range(20):
                x_interp = np.append(x_interp, x2)
                y_interp = np.append(y_interp, y2)
            #f len(dir)>0:
            #dir = np.append(dir, np.degrees(np.arctan2(ys,xs)))
            #else:
            #   dir = np.array([0])
        dir = np.degrees(np.arctan2(-np.gradient(y_interp), np.gradient(x_interp)))

        for i in range(len(trajectory['x'])):
            x = trajectory['x'][i]
            y = trajectory['y'][i]
            d = trajectory['dir'][i]

            ii = np.where(np.logical_and(x_interp==x, y_interp==y))
            dir[ii] = (d-1)*90.

        dir -= dir[21]

        plt.scatter(x_interp, y_interp, s=50, c=dir)
        plt.show()

        dist = [0]
        for i in range(1,len(x_interp)):
            d = np.sqrt((x_interp[i]-x_interp[i-1])**2 + (y_interp[i]-y_interp[i-1])**2)
            dist.append(dist[-1]+d)

        # Now get the z values for all of these points
        with open('trajectory_temp.xy', 'w') as f:
            for i in range(len(x_interp)):
                f.write(str(x_interp[i])+'\t'+str(y_interp[i])+'\n')
        cmd = 'gmt grdtrack trajectory_temp.xy -G'+self.dem.tif_path +" > trajectory_temp.xyz"
        print("Interpolating trajectory with "+str(dx)+" meter spacing...")
        subprocess.call([cmd], shell=True)

        xyz = np.loadtxt('trajectory_temp.xyz')

        trajectory_interp = {'x':xyz[:,0], 'y':xyz[:,1], 'z':xyz[:,2], 'dist':np.array(dist), 'dir':dir}

        # Finally, clean up
        os.remove("trajectory_temp.xy")
        os.remove("trajectory_temp.xyz")

        return trajectory_interp

if __name__ == "__main__":

    heightmap_path = "Heightmap.png"
    dem = DEM(heightmap_path, vmin=7, vmax=28)
    dem.geocode(bounds=[-255, 235, -265, 225]) # 500 x 500 approx in unreal
    pickle.dump(dem, open("dem.pkl", 'wb'))
    #dem = pickle.load(open("dem.pkl", 'rb'))

    rg = RoboGrid(dem)
    rg.compute_node_viewshed()
    pickle.dump(rg, open("robogrid.pkl", 'wb'))
    #rg = pickle.load(open('robogrid.pkl', 'rb'))



    '''
    fig, ax = plt.subplots()
    #ax.imshow(rg.dem.heightmap, extent=rg.dem.bounds)
    #rg.plot_shapefiles(ax=ax)
    #pickle.dump(rg, open("robogrid.pkl", 'wb'))

    #for i in rg.robogrid.keys():
    #    rx, ry = rg.robogrid[i]['UE_coords']
        #if i == 105:
            #plt.plot(rx, ry, 's', color='red', ms=10)
        #else:
        #plt.plot(rx, ry, 'or', ms=3)

    colors = ['m', 'orange', 'cyan', 'white']
    for i in range(4):
        nodes = rg.get_directional_nodes(node=105, d=i)
        for node in nodes:
            rx, ry = rg.robogrid[node]['UE_coords']
            plt.plot(rx, ry, 's', color=colors[i], ms=10)

    #plt.show()

    ## I now want to plot the average visibility for a specific node.
    #noi = 105 # Node of interest
    #rxs = []
    #rys = []
    #vs = []
    #for d in range(4):
    #    for node in rg.robogrid[noi]['viewshed'][d].keys():
    #        rx, ry = rg.robogrid[node]['UE_coords']
    #        v = rg.robogrid[noi]['viewshed'][d][node]
    #        rxs.append(rx)
    #        rys.append(ry)
    #        vs.append(v)
    #print(vs)
    #plt.scatter(rxs, rys, s=200, marker='s', c=vs, vmin=0, vmax=1, cmap='autumn')
    #plt.colorbar()

    # plot the viewshed ontop of the dem as a sanity check.
    #noi = int(np.median(list(rg.robogrid.keys())))
    #rx, ry = rg.robogrid[noi]['UE_coords']
    #viewshed = rg.dem.compute_viewshed(rx, ry)
    #plt.imshow(rg.dem.heightmap, extent=rg.dem.bounds)
    #plt.imshow(viewshed, extent=rg.dem.bounds, alpha=0.125, cmap='gray')
    #plt.plot(rx, ry, 'or', ms=5)

    plt.show()
    '''
