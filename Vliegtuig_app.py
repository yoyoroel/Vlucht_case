import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import folium
import plotly.express as px
import plotly.graph_objects as go
from geographiclib.geodesic import Geodesic
from streamlit_folium import st_folium
import numpy as np
import os

Kleur = px.colors.qualitative.Vivid

# Split the CSV into six parts

@st.cache_resource
def read_csv_safe(file_path):
    """Laad een CSV-bestand met caching."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path, low_memory=False)
    else:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the file is missing

# Laad de datasets met caching
Vluchtdata_part1 = read_csv_safe('Vluchtdata_part1.csv')
Vluchtdata_part2 = read_csv_safe('Vluchtdata_part2.csv')
Vluchtdata_part3 = read_csv_safe('Vluchtdata_part3.csv')
Vluchtdata_part4 = read_csv_safe('Vluchtdata_part4.csv')
Vluchtdata_part5 = read_csv_safe('Vluchtdata_part5.csv')
Vluchtdata_part6 = read_csv_safe('Vluchtdata_part6.csv')

# Merge the six parts into a single DataFrame
Vluchtdata = pd.concat([Vluchtdata_part1, Vluchtdata_part2, Vluchtdata_part3, Vluchtdata_part4, Vluchtdata_part5, Vluchtdata_part6], ignore_index=True)
Vluchtdata = Vluchtdata[Vluchtdata['Arr_airport'] != 'Montgomery Field']
conditions = [
    (Vluchtdata["Delay_minutes"] < -10),
    (Vluchtdata["Delay_minutes"] >= -10) & (Vluchtdata["Delay_minutes"] < 15),
    (Vluchtdata["Delay_minutes"] >= 15)
]
choices = ["Early", "On Time", "Delayed"]
Vluchtdata["Flight_status"] = np.select(conditions, choices, default="Unknown")

Vluchtdata['Actual_dt'] = pd.to_datetime(Vluchtdata['Date'] + ' ' + Vluchtdata['Actual_time'], format='%d/%m/%Y %H:%M:%S')
Vluchtdata["Weekday"] = Vluchtdata["Actual_dt"].dt.day_name()
Vluchtdata["Month"] = Vluchtdata["Actual_dt"].dt.month_name()
Vluchtdata["Year"] = Vluchtdata["Actual_dt"].dt.year
Vluchtdata["Hour"] = Vluchtdata["Actual_dt"].dt.hour
Vluchtdata['Date'] = Vluchtdata['Date'].astype(str).str.strip()
Vluchtdata['Date'] = pd.to_datetime(Vluchtdata['Date'], format='%d/%m/%Y', errors='coerce')

Zurich = [47.464699,8.549170]

Swiss = Vluchtdata.dropna(subset=['Arr_lat', 'Arr_lng'])

o = folium.Map(location=Zurich, zoom_start=2, tiles = None)
folium.TileLayer(tiles="OpenStreetMap", name="Blauwe kaart").add_to(o)
folium.TileLayer(tiles="cartodbpositron", name="Grijze kaart").add_to(o)

Swiss_unique = Swiss.drop_duplicates(subset=['Arr_airport'])

continent_colors = {continent: Kleur[i % len(Kleur)] for i, continent in enumerate(Swiss_unique['Continent_arr'].unique())}


def interpolate_great_circle(start, end, n_points=30):
    geod = Geodesic.WGS84
    l = geod.InverseLine(start[0], start[1], end[0], end[1])
    points = []
    for i in range(n_points + 1):
        s = l.s13 * i / n_points
        pos = l.Position(s)
        points.append((pos['lat2'], pos['lon2']))
    return points


for continent in Swiss_unique['Continent_arr'].unique():
    continent_layer = folium.FeatureGroup(name=continent)
    continent_data = Swiss_unique[Swiss_unique['Continent_arr'] == continent]
    
    for i in continent_data.index:
        destination = [continent_data.loc[i, 'Arr_lat'], continent_data.loc[i, 'Arr_lng']]
        color = continent_colors.get(continent, Kleur[0])
        
        folium.Circle(
            location=destination,
            color=color,
            popup=continent_data.loc[i, 'Arr_airport']
        ).add_to(continent_layer)
        
        # Hier genereren we de kromme lijn
        curve_points = interpolate_great_circle(Zurich, destination, n_points=50)
        
        folium.PolyLine(
            locations=curve_points,
            color=color,
            weight=2
        ).add_to(continent_layer)
    
    continent_layer.add_to(o)


folium.LayerControl().add_to(o)

folium.Circle(location = Zurich, color = Kleur[-1]).add_to(o)

Zurich = [47.464699,8.549170]

Swiss = Vluchtdata[(Vluchtdata['airline_code'] == 'LX') & (Vluchtdata['LSV'] == 'Outbound')]
Swiss = Swiss.dropna(subset=['Arr_lat', 'Arr_lng'])

n = folium.Map(location=Zurich, zoom_start=2, tiles = None)
folium.TileLayer(tiles="OpenStreetMap", name="Blauwe kaart").add_to(n)
folium.TileLayer(tiles="cartodbpositron", name="Grijze kaart").add_to(n)

Swiss_unique = Swiss.drop_duplicates(subset=['Arr_airport'])

continent_colors = {continent: Kleur[i % len(Kleur)] for i, continent in enumerate(Swiss_unique['Continent_arr'].unique())}

def interpolate_great_circle(start, end, n_points=30):
    geod = Geodesic.WGS84
    l = geod.InverseLine(start[0], start[1], end[0], end[1])
    points = []
    for i in range(n_points + 1):
        s = l.s13 * i / n_points
        pos = l.Position(s)
        points.append((pos['lat2'], pos['lon2']))
    return points


for continent in Swiss_unique['Continent_arr'].unique():
    continent_layer = folium.FeatureGroup(name=continent)
    continent_data = Swiss_unique[Swiss_unique['Continent_arr'] == continent]
    
    for i in continent_data.index:
        destination = [continent_data.loc[i, 'Arr_lat'], continent_data.loc[i, 'Arr_lng']]
        color = continent_colors.get(continent, Kleur[0])
        
        folium.Circle(
            location=destination,
            color=color,
            popup=continent_data.loc[i, 'Arr_airport']
        ).add_to(continent_layer)
        
        # Hier genereren we de kromme lijn
        curve_points = interpolate_great_circle(Zurich, destination, n_points=50)
        
        folium.PolyLine(
            locations=curve_points,
            color=color,
            weight=2
        ).add_to(continent_layer)
    
    continent_layer.add_to(n)


folium.LayerControl().add_to(n)

folium.Circle(location = Zurich, color = Kleur[-1]).add_to(n)

# Voeg een navigatiemenu toe
pagina = st.sidebar.radio(
    "Selecteer een pagina",
    ["Netwerk Zurich", "Netwerk Swiss", "Live Vluchtdata", "Aircraft Ground Count Analysis", "Vertraging per route", "Vertraging voorspellen"]
)

if pagina == "Netwerk Zurich":
    # Code voor Kaart O
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title('Netwerk Zurich Airport')
        st.write("Deze kaart toont de bestemmingen vanuit Zurich, inclusief de mogelijkheid om continenten aan of uit te zetten en details van luchthavens te bekijken door op de punten te klikken.")

    with col2:
        st.image("Zurich Logo.png", width=300)
    
    st.components.v1.html(o._repr_html_(), height=600)

elif pagina == "Netwerk Swiss":
    # Code voor Kaart N
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title('Netwerk Swiss')
        st.write("Deze kaart toont de bestemmingen van Swiss vluchten vanuit Zurich, inclusief de mogelijkheid om continenten aan of uit te zetten en details van luchthavens te bekijken door op de punten te klikken.")

    with col2:
        st.image("Swiss Logo.png", width=300)

    st.components.v1.html(n._repr_html_(), height=600)

elif pagina == "Live Vluchtdata":
    pd.set_option('display.max_columns', None)
    st.title('Live Vluchtdata')   

    # Data import
    flightdata1 = pd.read_excel('case3/30Flight 1.xlsx')
    flightdata2 = pd.read_excel('case3/30Flight 2.xlsx')
    flightdata3 = pd.read_excel('case3/30Flight 3.xlsx')
    flightdata4 = pd.read_excel('case3/30Flight 4.xlsx')
    flightdata5 = pd.read_excel('case3/30Flight 5.xlsx')
    flightdata6 = pd.read_excel('case3/30Flight 6.xlsx')
    flightdata7 = pd.read_excel('case3/30Flight 7.xlsx')

    # Combine all flight data into a dictionary
    flightdata_options = {
        'Flight 1': flightdata1,
        'Flight 2': flightdata2,
        'Flight 3': flightdata3,
        'Flight 4': flightdata4,
        'Flight 5': flightdata5,
        'Flight 6': flightdata6,
        'Flight 7': flightdata7
    }

    # Create a select box for flight options
    selected_flight = st.selectbox('Select a flight', list(flightdata_options.keys()))

    # Get the selected flight data
    flightdata = flightdata_options[selected_flight]

    # kleur voor de hoogte
    def flkleur(hoogte):
        if hoogte > 40000:
            return 'purple'
        elif hoogte > 30000:
            return 'blue'
        elif hoogte > 20000:
            return 'dodgerblue'
        elif hoogte > 10000:
            return 'lawngreen'
        elif hoogte > 8000:
            return 'greenyellow'
        elif hoogte > 4000:
            return 'yellow'
        elif hoogte > 2000:
            return 'gold'

    # Add a slider to select a range of time in minutes
    start_time, end_time = st.slider(
        'Select time range (minutes)',
        min_value=0,
        max_value=int(flightdata['Time (secs)'].max() // 60),
        value=(0, int(flightdata['Time (secs)'].max() // 60))
    )

    # Convert the selected time range back to seconds
    start_time *= 60
    end_time *= 60

    # Filter the flight data based on the selected time range
    filtered_flightdata = flightdata[(flightdata['Time (secs)'] >= start_time) & (flightdata['Time (secs)'] <= end_time)]

    # Ensure 'TRUE AIRSPEED (derived)' is numeric
    filtered_flightdata['TRUE AIRSPEED (derived)'] = pd.to_numeric(filtered_flightdata['TRUE AIRSPEED (derived)'], errors='coerce')

    # Function to create and display the map
    def create_map(flightdata):
        m = folium.Map(location=[47.9027336, 1.9086066], zoom_start=5)
        folium.TileLayer(tiles="cartodbpositron", name="Grijze kaart").add_to(m)

        for i in flightdata.index:
            if not pd.isna(flightdata.loc[i, "[3d Latitude]"]) and not pd.isna(flightdata.loc[i, "[3d Longitude]"]):
                folium.Circle(
                    location=[flightdata.loc[i, "[3d Latitude]"], flightdata.loc[i, "[3d Longitude]"]],
                    popup=f"Heading: {str(flightdata.loc[i, '[3d Heading]'])}, Speed: {str(flightdata.loc[i, 'TRUE AIRSPEED (derived)'])}, Hoogte: {str(flightdata.loc[i, '[3d Altitude Ft]'])}",
                    color=flkleur(flightdata.loc[i, "[3d Altitude Ft]"]),
                    fill=True,
                    fill_color=flkleur(flightdata.loc[i, "[3d Altitude Ft]"]),
                    radius=500
                ).add_to(m)

        st_folium(m, width=1000, height=500)

    # Create and display the map for the filtered flight data
    create_map(filtered_flightdata)

    # Calculate the average data
    gemiddelde_airspeed = filtered_flightdata['TRUE AIRSPEED (derived)'].mean(skipna=True)
    min_airspeed = filtered_flightdata['TRUE AIRSPEED (derived)'].min(skipna=True)
    max_airspeed = filtered_flightdata['TRUE AIRSPEED (derived)'].max(skipna=True)
    gemiddelde_stijging = filtered_flightdata['[3d Altitude Ft]'].diff().mean(skipna=True)
    min_stijging = filtered_flightdata['[3d Altitude Ft]'].diff().min(skipna=True)
    max_stijging = filtered_flightdata['[3d Altitude Ft]'].diff().max(skipna=True)
    gemiddelde_hoogte = filtered_flightdata['[3d Altitude Ft]'].mean(skipna=True)
    min_hoogte = filtered_flightdata['[3d Altitude Ft]'].min(skipna=True)
    max_hoogte = filtered_flightdata['[3d Altitude Ft]'].max(skipna=True)

    # Maak drie kolommen
    col1, col2, col3 = st.columns(3)

    # Toon de waarden in de kolommen
    with col1:
        st.metric(label="Gemiddelde Snelheid (knots)", value=f"{gemiddelde_airspeed:.2f}")
    with col2:
        st.metric(label="Minimale Snelheid (knots)", value=f"{min_airspeed:.2f}")
    with col3:
        st.metric(label="Maximale Snelheid (knots)", value=f"{max_airspeed:.2f}")

    # Maak drie kolommen
    col1, col2, col3 = st.columns(3)

    # Toon de waarden in de kolommen
    with col1:
        st.metric(label="Gemiddelde Hoogte (ft)", value=f"{gemiddelde_hoogte:.2f}")
    with col2:
        st.metric(label="Minimale Hoogte (ft)", value=f"{min_hoogte:.2f}")
    with col3:
        st.metric(label="Maximale Hoogte (ft)", value=f"{max_hoogte:.2f}")

    # Maak drie kolommen
    col1, col2, col3 = st.columns(3)

    # Toon de waarden in de kolommen
    with col1:
        st.metric(label="Gemiddelde Stijging (ft)", value=f"{gemiddelde_stijging:.2f}")
    with col2:
        st.metric(label="Minimale Stijging (ft)", value=f"{min_stijging:.2f}")
    with col3:
        st.metric(label="Maximale Stijging (ft)", value=f"{max_stijging:.2f}")

    # Plot van snelheid en hoogte als lijn om de twee te vergelijken
    fig = px.line(
        filtered_flightdata,
        x=filtered_flightdata.index,
        y=['[3d Altitude Ft]', 'TRUE AIRSPEED (derived)'],
        labels={'value': 'Waarde', 'index': 'Tijd (index)', 'variable': 'Variabele'},
        title='Vergelijking van Hoogte en Snelheid',
        color_discrete_sequence=[Kleur[0], Kleur[1]]  # Gebruik kleuren uit de Kleur-lijst
    )

    # Pas de y-as aan zodat de schaal van TRUE AIRSPEED (derived) beter past
    fig.update_layout(
        yaxis=dict(
            title='Hoogte (ft)',
            side='left'
        ),
        yaxis2=dict(
            title='Snelheid (knots)',
            overlaying='y',
            side='right'
        ),
        legend_title_text='Metingen'
    )

    # Pas de namen in de legende aan
    fig.for_each_trace(lambda t: t.update(name='Hoogte (ft)' if t.name == '[3d Altitude Ft]' else 'Snelheid (knots)'))

    # Voeg de tweede y-as toe voor TRUE AIRSPEED (derived)
    fig.for_each_trace(lambda t: t.update(yaxis='y2') if t.name == 'Snelheid (knots)' else None)

    st.plotly_chart(fig)  

elif pagina == 'Aircraft Ground Count Analysis':

    df = Vluchtdata

    # Streamlit UI
    st.title("Aircraft Ground Count Analysis")

    # Zorg ervoor dat 'Actual_dt' in datetime-formaat is
    df["Actual_dt"] = pd.to_datetime(df["Actual_dt"])
    df["Date"] = df["Actual_dt"].dt.date  # Unieke datums zonder tijd

    # 📅 Maak een kalender-widget
    selected_date = st.date_input("Selecteer een datum", min_value=min(df["Date"]), max_value=max(df["Date"]))

    # Filter de data voor de geselecteerde datum
    df_day = df.loc[df["Date"] == selected_date].copy()

    # Alleen doorgaan als er data is voor de geselecteerde datum
    if not df_day.empty:
        df_outbound = df_day[df_day["LSV"] == "Outbound"].copy()
        df_inbound = df_day[df_day["LSV"] == "Inbound"].copy()

        df_outbound = df_outbound.sort_values(by="Actual_dt")
        df_inbound = df_inbound.sort_values(by="Actual_dt")

        start_time = df_day["Actual_dt"].min()
        end_time = df_day["Actual_dt"].max()

        intervals = pd.date_range(start=start_time, end=end_time, freq="10min")

        ground_counts = []
        on_ground = 40  # Beginwaarde (kan worden aangepast)

        for interval in intervals:
            on_ground += (df_inbound["Actual_dt"] <= interval).sum()
            df_inbound = df_inbound[df_inbound["Actual_dt"] > interval]

            on_ground -= (df_outbound["Actual_dt"] <= interval).sum()
            df_outbound = df_outbound[df_outbound["Actual_dt"] > interval]

            ground_counts.append(on_ground)

        # Maak de grafiek
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=intervals,
            y=ground_counts,
            mode='lines+markers',
            name=str(selected_date),
            line=dict(color=Kleur[1]),
            marker=dict(color=Kleur[1]),
        ))

        fig.update_layout(
            title=f"Aircraft on the ground on {selected_date}",
            xaxis_title="Time",
            yaxis_title="Number of aircraft",
        )

        # Toon de interactieve grafiek in Streamlit
        st.plotly_chart(fig)
    else:
        st.warning("Geen data beschikbaar voor deze datum. Probeer een andere.")


    # Streamlit UI
    st.title("Flight Status Verdeling per Airline")

    # Alleen airlines met 100 of meer vluchten behouden
    df_filtered = df.groupby('airline_name').filter(lambda x: len(x) >= 100)

    # Absolute aantallen
    counts = df_filtered.groupby(["airline_name", "Flight_status"])["Delay_minutes"].count().reset_index(name='count')

    # Totale vluchten per airline
    total_counts = counts.groupby('airline_name')['count'].transform('sum')
    # Relatieve aantallen (percentages)
    counts['percentage'] = (counts['count'] / total_counts) * 100

    # Unieke airlines ophalen
    airlines = counts['airline_name'].unique()

    # 🔎 Zoekfunctie invoerveld
    search_query = st.text_input("Zoek een airline:", "")

    # Filter airlines op basis van de zoekopdracht (case insensitive)
    filtered_airlines = [a for a in airlines if search_query.lower() in a.lower()]

    if not filtered_airlines:
        st.warning("Geen resultaten gevonden. Probeer een andere zoekterm.")
    else:
        # Pak de eerste match als standaard airline
        selected_airline = filtered_airlines[0]

        # Data filteren voor geselecteerde airline
        subset = counts[counts['airline_name'] == selected_airline]

        # Maak een figuur
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=subset['Flight_status'],
                y=subset['percentage'],
                name=selected_airline,
                marker_color=Kleur[1]
            )
        )

        # Layout instellen
        fig.update_layout(
            title=f"Flight status verdeling voor {selected_airline}",
            xaxis_title="Flight Status",
            yaxis_title="Percentage (%)",
            showlegend=False,
        )

        # Zorg ervoor dat de categorieën in een vaste volgorde blijven
        fig.update_xaxes(categoryorder="array", categoryarray=["Early", "On Time", "Delayed"])

        # Toon de interactieve grafiek in Streamlit
        st.plotly_chart(fig)

        # Bepaal de airline met de hoogste vertraging
        idx = counts.groupby("airline_name")["percentage"].idxmax()
        highest_percentage = counts.loc[idx]

        st.write("### Airlines met de hoogste vertraging per categorie")
        st.dataframe(highest_percentage)



    df["Actual_dt"] = pd.to_datetime(df["Actual_dt"])
    df["Hour"] = df["Actual_dt"].dt.hour
    df["Date"] = df["Actual_dt"].dt.date

    # Groepeer per Date, Hour en LSV
    grouped_data = df.groupby(["Date", "Hour", "LSV"]).size().reset_index(name="Count")

    # Streamlit interface
    st.title("Lijngrafiek per uur op basis van geselecteerde datum")

    # Kalender-widget om een dag te kiezen
    selected_date = st.date_input("Selecteer een datum", min_value=df["Date"].min(), max_value=df["Date"].max(), key='unique_date_picker')

    # Filter data op de geselecteerde datum
    filtered_df = grouped_data[grouped_data["Date"] == selected_date]

    # Voeg een checkbox toe om de vertraging data wel of niet te tonen
    show_delay_data = st.checkbox("Toon gemiddelde vertraging", value=True)

    # Bereken het gemiddelde aantal minuten vertraging per uur
    df["Hour_interval"] = df["Actual_dt"].dt.floor("h")
    delay_data = df.groupby(["Date", "Hour_interval"])["Delay_minutes"].mean().reset_index(name="Avg_Delay")

    # Voeg de gemiddelde vertraging toe aan de gefilterde data
    delay_filtered = delay_data[delay_data["Date"] == selected_date]

    # Plot de data met Plotly
    fig = px.line(filtered_df, x="Hour", y="Count", color="LSV", markers=True,
                title=f"Aantal inbound en outbound per uur op {selected_date}",
                labels={"Hour": "Uur", "Count": "Aantal vluchten", "LSV": "Legenda"},
                color_discrete_map={"Inbound":Kleur[0], "Outbound":Kleur[1]})

    # Voeg de lijn voor gemiddelde vertraging toe als de checkbox is aangevinkt
    if show_delay_data:
        # Bereken de maximale en minimale vertraging per uur
        delay_data_stats = df.groupby(["Date", "Hour_interval"])["Delay_minutes"].agg(["mean", "max", "min"]).reset_index()
        delay_filtered_stats = delay_data_stats[delay_data_stats["Date"] == selected_date]

        fig.add_trace(go.Scatter(
            x=delay_filtered_stats["Hour_interval"].dt.hour,
            y=delay_filtered_stats["mean"],
            mode="lines+markers",
            name="Gemiddelde vertraging (min)",
            yaxis="y2",
            line=dict(color=Kleur[2]),
            hovertemplate=(
                "Gemiddelde vertraging: %{y:.2f} min<br>"
                "Uur: %{x} uur"
            ),
            customdata=delay_filtered_stats[["max", "min"]].values
        ))

        # Zorg ervoor dat de y-as gidslijnen gelijk blijven
        yaxis_range = [min(delay_filtered_stats["mean"])-5, max(filtered_df["Count"].max(), delay_filtered_stats["mean"].max()) * 1.1]
        fig.update_layout(
            yaxis=dict(
                title='Aantal vluchten',
                side='left',
                range=yaxis_range
            ),
            yaxis2=dict(
                title='Gemiddelde vertraging (min)',
                overlaying='y',
                side='right',
                range=yaxis_range
            )
        )
    else:
        # Maak de y-as variabel als de delaylijn uit staat
        fig.update_layout(
            yaxis=dict(
                title='Aantal vluchten',
                side='left',
            )
        )

    # Plaats de legenda altijd onder de grafiek
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Toon de grafiek in Streamlit
    st.plotly_chart(fig)

elif pagina == 'Vertraging per route':
    import pandas as pd
    import folium
    import streamlit as st
    from pyproj import Geod
    import branca.colormap as cm

    def interpolate_great_circle(start, end, n_points=30):
        geod = Geod(ellps="WGS84")
        intermediate_points = geod.npts(start[1], start[0], end[1], end[0], n_points - 2)
        points = [start] + [(lat, lon) for lon, lat in intermediate_points] + [end]
        return points

    st.title("Vluchtvertragingen Visualisatie")

    gemiddelde_vertraging = Vluchtdata.groupby([ 
        'Outbound_airport', 'Inbound_airport', 'LSV', 'Dep_lat', 'Dep_lng', 'Arr_lat', 'Arr_lng'
    ])['Delay_minutes'].mean().reset_index()

    lsv_selectie = st.radio("Selecteer type vlucht:", ["Outbound", "Inbound"])
    vertraging_data = gemiddelde_vertraging[gemiddelde_vertraging['LSV'] == lsv_selectie]

    n_points = st.slider("Aantal interpolatiepunten (boog-resolutie)", min_value=5, max_value=100, value=30)

    zurich_lat = 47.464699
    zurich_lng = 8.549170

    # Definieer de folium-kaart 'q'
    q = folium.Map(location=[0, 0], zoom_start=2)

    # Pas de colormap aan zodat 15 minuten oranje is
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'orange', 'red'],
                                vmin=vertraging_data['Delay_minutes'].min(),
                                vmax=vertraging_data['Delay_minutes'].max())
    colormap.caption = 'Gemiddelde Vertraging (minuten)'
    colormap.add_to(q)

    folium.Circle(
        location=[zurich_lat, zurich_lng],
        radius=5000,
        color='blue',
        fill=True,
        fill_color='blue',
        popup='Zurich'
    ).add_to(q)

    for _, row in vertraging_data.iterrows():
        delay = row['Delay_minutes']
        color = colormap(delay)
        
        if lsv_selectie == 'Outbound':
            start_point = [zurich_lat, zurich_lng]
            destination = [row['Arr_lat'], row['Arr_lng']]
            popup_text = f"{row['Inbound_airport']} - Vertraging: {delay:.1f} min"
        else:
            start_point = [row['Dep_lat'], row['Dep_lng']]
            destination = [zurich_lat, zurich_lng]
            popup_text = f"{row['Outbound_airport']} - Vertraging: {delay:.1f} min"

        # Marker afhankelijk van richting
        if lsv_selectie == 'Inbound':
            folium.Circle(
                location=start_point,
                radius=3000,
                color=color,
                fill=True,
                fill_color=color,
                popup=popup_text
            ).add_to(q)
        else:
            folium.Circle(
                location=destination,
                radius=3000,
                color=color,
                fill=True,
                fill_color=color,
                popup=popup_text
            ).add_to(q)

        # Interpolatie van de grote cirkel
        gc_points = interpolate_great_circle(start_point, destination, n_points=n_points)

        # Maak een PolyLine van de grote cirkel
        polyline = folium.PolyLine(
            locations=gc_points,
            color=color,
            weight=2
        )

        # Voeg popup toe aan de lijn (PolyLine)
        polyline.add_child(folium.Popup(popup_text))  # Voeg de popup toe aan de PolyLine

        # Voeg de lijn toe aan de kaart
        polyline.add_to(q)
            # Kaart tonen
    st.components.v1.html(q._repr_html_(), height=600)
    
elif pagina == "Vertraging voorspellen":
    df_2019 = Vluchtdata[Vluchtdata['Actual_dt'].dt.year == 2019]

    #Normaliseer de 'Flight_status'-kolom
    df_2019.loc[:, 'Flight_status'] = df_2019['Flight_status'].str.strip().str.lower()

    #Tel het totaal aantal vluchten per dag
    total_flights_per_day = df_2019.groupby('Date').size().reset_index(name='Total Flights')

    #Tel het aantal vertraagde vluchten per dag (Flight_status == "delayed")
    delayed_flights_per_day = df_2019[df_2019['Flight_status'] == "delayed"].groupby('Date').size().reset_index(name='Delayed Flights')

    #Merge de dataframes op 'Date'
    merged_df = total_flights_per_day.merge(delayed_flights_per_day, on='Date', how='left').fillna(0)

    #**Toon data-informatie in Streamlit**
    st.subheader("Correlatie tussen aantal vluchten en vertraagde vluchten")

    # Voeg een checkbox toe om de trendlijn aan of uit te zetten
    show_trendline = st.checkbox("Toon trendlijn", value=True)

    # ⬇️ Maak een interactieve scatterplot met of zonder trendlijn
    fig = px.scatter(
        merged_df,
        x='Total Flights',
        y='Delayed Flights',
        title="Correlatie tussen totaal aantal vluchten en vertraagde vluchten per dag (2019)",
        labels={
            'Total Flights': 'Totaal aantal vluchten per dag',
            'Delayed Flights': 'Aantal vertraagde vluchten per dag'
        },
        trendline="ols" if show_trendline else None,
        trendline_color_override='red' if show_trendline else None
    )

    # Toon de scatterplot in Streamlit
    st.plotly_chart(fig)

    st.subheader("Lijngrafiek totaal en vertraagd aantal vluchten per maand")
    df_2019_2 = Vluchtdata[Vluchtdata["Year"] == 2019]

    total_flights = df_2019_2.groupby(['Month', 'Hour']).size().reset_index(name='Total Flights')

    # ⬇️ Tel het aantal vertraagde vluchten per maand en uur
    delayed_flights = df_2019_2[df_2019_2['Flight_status'] == "Delayed"].groupby(['Month', 'Hour']).size().reset_index(name='Delayed Flights')

    # ⬇️ Merge de dataframes
    merged_df_2 = total_flights.merge(delayed_flights, on=['Month', 'Hour'], how='left').fillna(0)

    # ⬇️ Streamlit UI
    st.title("Vluchten en Vertragingen per Uur (2019)")

    # ⬇️ Selecteer een maand
    month_options = merged_df_2['Month'].unique()
    selected_month = st.selectbox("Kies een maand:", month_options)

    # ⬇️ Filter de dataset voor de gekozen maand
    month_df = merged_df_2[merged_df_2['Month'] == selected_month]

    # ⬇️ Maak de Plotly-figuur
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=month_df['Hour'], y=month_df['Total Flights'], 
        mode='lines+markers', name="Total Flights"
    ))
    fig.add_trace(go.Scatter(
        x=month_df['Hour'], y=month_df['Delayed Flights'], 
        mode='lines+markers', name="Delayed Flights"
    ))

    fig.update_layout(
        title=f"Aantal vluchten en vertraagde vluchten per uur (2019) - Maand {selected_month}",
        xaxis_title="Uur van de dag",
        yaxis_title="Aantal vluchten",
        template="plotly_white"
    )

    # ⬇️ Toon de grafiek in Streamlit
    st.plotly_chart(fig)


    st.subheader("Top 10 airlines met meeste vertraging")
    df_filtered = Vluchtdata.groupby('airline_name').filter(lambda x: len(x) >= 100)

    ### --- ANALYSE PER AIRLINE --- ###

    # Split inbound en outbound voor airlines
    inbound_flights_airline = df_filtered[df_filtered['LSV'] == 'Inbound']
    outbound_flights_airline = df_filtered[df_filtered['LSV'] == 'Outbound']

    # Bereken vertraging per airline voor inbound vluchten
    inbound_total_airline = inbound_flights_airline.groupby('airline_name').size().reset_index(name='total_flights')
    inbound_delayed_airline = inbound_flights_airline[inbound_flights_airline['Flight_status'] == 'Delayed'].groupby('airline_name').size().reset_index(name='delayed_count')
    inbound_percentage_airline = pd.merge(inbound_delayed_airline, inbound_total_airline, on='airline_name', how='right').fillna(0)
    inbound_percentage_airline['delay_ratio'] = (inbound_percentage_airline['delayed_count'] / inbound_percentage_airline['total_flights']) * 100
    top_10_inbound_airline = inbound_percentage_airline.sort_values(by='delay_ratio', ascending=False).head(10)

    # Bereken vertraging per airline voor outbound vluchten
    outbound_total_airline = outbound_flights_airline.groupby('airline_name').size().reset_index(name='total_flights')
    outbound_delayed_airline = outbound_flights_airline[outbound_flights_airline['Flight_status'] == 'Delayed'].groupby('airline_name').size().reset_index(name='delayed_count')
    outbound_percentage_airline = pd.merge(outbound_delayed_airline, outbound_total_airline, on='airline_name', how='right').fillna(0)
    outbound_percentage_airline['delay_ratio'] = (outbound_percentage_airline['delayed_count'] / outbound_percentage_airline['total_flights']) * 100
    top_10_outbound_airline = outbound_percentage_airline.sort_values(by='delay_ratio', ascending=False).head(10)

    ### --- VISUALISATIE --- ###

    # Airlines inbound vertraging
    fig_inbound_airline = px.bar(
        top_10_inbound_airline,
        x='airline_name',
        y='delay_ratio',
        title='Top 10 Airlines met Hoogste Vertraging (Inbound)',
        labels={'airline_name': 'Airline', 'delay_ratio': 'Vertraging (%)'},
        color='delay_ratio',
        color_continuous_scale='Blues'

    )

    # Airlines outbound vertraging
    fig_outbound_airline = px.bar(
        top_10_outbound_airline,
        x='airline_name',
        y='delay_ratio',
        title='Top 10 Airlines met Hoogste Vertraging (Outbound)',
        labels={'airline_name': 'Airline', 'delay_ratio': 'Vertraging (%)'},
        color='delay_ratio',
        color_continuous_scale='Reds'

    )

    # Toon alle plots in Streamlit
    st.plotly_chart(fig_inbound_airline)
    st.plotly_chart(fig_outbound_airline)

    st.subheader("Boxplot van vertraging per concourse")
    df_outbound_box = Vluchtdata[Vluchtdata["LSV"] == "Outbound"]

    # Filter vertragingen tussen -15 en 60 minuten
    df_filter_box = df_outbound_box[(df_outbound_box["Delay_minutes"] > -15) & (df_outbound_box["Delay_minutes"] < 60)]

    fig = px.box(
    df_filter_box,
    x="Concourse",
    y="Delay_minutes",
    title="Vertragingen per Concourse (Outbound)",
    labels={"Concourse": "Concourse", "Delay_minutes": "Vertraging in Minuten"},
    )
    st.plotly_chart(fig)
